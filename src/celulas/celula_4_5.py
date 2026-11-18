# @title
# Célula 4.5: Sensibilidade da Amostragem do GPTCast — na validação de 2024, compara a forma de sortear os tokens
# (padrão do forecast(): temperatura 1 e sem top-k; top-k 50 e 10; temperatura 0,7; escolha gulosa; e a média de
# 8 sorteios) no Fine-Tuned e no Zero-Shot, contra a persistência. Escolhe a configuração em 2024 para depois
# aplicá-la no teste de 2025, sem usar 2025 na escolha. Retomável por janela; não altera modelos nem checkpoints.
import contextlib
import glob
import hashlib
import io
import json
import math
import os
import textwrap
import time
import warnings

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gptcast.utils.converters import dbz_to_rainfall

# ------------------------------------------------------------------ 1. parâmetros
N_JANELAS = 240          # janelas da validação de 2024 sorteadas (semente fixa); cerca de 20 s por janela na G4
N_MEMBROS = 8            # sorteios da média (gerados juntos, num só lote na GPU)
TEMPO_MAX_MIN = 150      # orçamento desta execução; o que ficar pronto é salvo e a próxima continua
SAL_STRIDE = 1
SEED = 42
STEP_MIN = 30
THRESHOLDS = (0.5, 1.0, 5.0, 10.0)
N_BOOT = 2000
# nome -> (temperatura, top_k, membros)
CONFIGS = {
    'Padrão (T=1, sem top-k)': (1.0, None, 1),
    'top-k 50': (1.0, 50, 1),
    'top-k 10': (1.0, 10, 1),
    'Temperatura 0,7': (0.7, None, 1),
    'Gulosa (top-k 1)': (1.0, 1, 1),
    f'Média de {N_MEMBROS} sorteios (T=1)': (1.0, None, N_MEMBROS),
}
PADRAO = 'Padrão (T=1, sem top-k)'
ENV_TAG = globals().get('ENV_TAG', 'Ambiente_Ativo')
ENV_LABEL = globals().get('ENV_LABEL', 'Hardware Ativo')

faltam = [n for n in ('gpt8', 'gpt8_zeroshot', 'md_rmf', 'RESULTS_ENV_DIR', 'CONTEXT_STEPS', 'FORECAST_STEPS')
          if n not in globals()]
if faltam:
  raise RuntimeError(f'Rode antes a Célula 4. Faltam: {", ".join(faltam)}')
if not torch.cuda.is_available():
  raise RuntimeError('A Célula 4.5 gera milhares de previsões do GPTCast: ative a GPU.')
DEVICE = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
gpt8, gpt8_zeroshot = gpt8.to(DEVICE).eval(), gpt8_zeroshot.to(DEVICE).eval()
MODELOS = {'Fine-Tuned': gpt8, 'Zero-Shot': gpt8_zeroshot}
C, F = int(CONTEXT_STEPS), int(FORECAST_STEPS)
WS = int(math.sqrt(gpt8.transformer.config.block_size // 8))     # janela espacial do GPT, como no forecast()

ds = md_rmf.splits['val']
rng0 = np.random.default_rng(SEED)
SEL = np.sort(rng0.choice(len(ds.sequences_dbz), size=min(N_JANELAS, len(ds.sequences_dbz)), replace=False))
ts_sel = np.asarray(ds.timestamps)[SEL]


def _rain(dbz):
  return np.clip(dbz_to_rainfall(np.clip(np.asarray(dbz, dtype=np.float64), 0.0, 60.0)), 0.0, 205.0).astype(np.float32)


def _digest(model):
  h = hashlib.sha256()
  for k, v in sorted(model.state_dict().items()):
    if k.startswith('transformer.'):
      h.update(k.encode())
      h.update(v.detach().float().cpu().numpy().tobytes())
  return h.hexdigest()[:10]


TAG = hashlib.sha256(json.dumps([list(CONFIGS.items()), SEL.tolist(), SEED]).encode()).hexdigest()[:8]
CACHE = os.path.join(RESULTS_ENV_DIR, f'amostragem_val2024_{_digest(gpt8)}_{TAG}')
os.makedirs(CACHE, exist_ok=True)


# ------------------------------------------------------------------ 2. gerador com controle da amostragem
# Reproduz o predict_sequence() do pacote gptcast (o mesmo que o forecast() chama): contexto de 7 quadros empilhado
# e codificado pela VQGAN em float32, tokens do próximo quadro gerados pelo Transformer em bf16 dentro da janela
# espacial WS x WS, cada quadro previsto decodificado isolado. A diferença é expor temperatura e top-k e gerar
# vários sorteios de uma vez (o código do GPTCast já aceita a dimensão de lote).
# O predict_next_frame_indices() do pacote grava cada token com predicted_indices[:, i, j] = res, e o
# predict_next_index() devolve res com forma (lote, 1). Com lote 1 isso passa, mas com vários sorteios no mesmo lote
# dá erro de forma. Durante a geração em lote, o predict_next_index() deste modelo é envolvido para devolver (lote,),
# sem mudar o sorteio; ao sair, o método original volta (as outras células não são afetadas).
@contextlib.contextmanager
def _lote(model, n):
  if n == 1:
    yield
    return
  original = model.predict_next_index
  model.predict_next_index = lambda contexto, temperature=1.0, top_k=None: original(contexto, temperature, top_k)[:, 0]
  try:
    yield
  finally:
    del model.predict_next_index


@torch.inference_mode()
def gerar(model, ctx_dbz, temperatura, top_k, n, seed):
  torch.manual_seed(seed)
  x = torch.tensor(np.clip(ctx_dbz, 0, 60) / 30.0 - 1.0, dtype=torch.float32, device=DEVICE)   # (s, h, w)
  s = x.shape[0]
  xs = einops.rearrange(x, 's h w -> (s h) w')[None, None].contiguous()
  quant, idx = model.encode_to_z(xs)
  _, c_idx = model.encode_to_c(xs)
  qs = quant.shape
  ind = idx.reshape(qs[0], s, qs[2] // s, qs[3]).repeat(n, 1, 1, 1)
  c_idx = c_idx.repeat(n, 1)
  ih, iw = ind.shape[2], ind.shape[3]
  quadros = []
  for _ in range(F):
    with torch.autocast('cuda', dtype=torch.bfloat16), _lote(model, n):
      p = model.predict_next_frame_indices(ind[:, -s:], c_idx, WS, temperature=temperatura, top_k=top_k,
                                           show_progress=False)
    quadros.append(model.decode_to_img(p.reshape(n, -1), torch.Size([n, qs[1], ih, iw])))
    ind = torch.cat((ind, p[:, None]), dim=1)
  y = torch.cat(quadros, dim=1).float().cpu().numpy().clip(-1, 1)                             # (n, F, h, w)
  return _rain((y + 1.0) * 30.0)


@contextlib.contextmanager
def _silencio():
  with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
    warnings.simplefilter('ignore')
    yield


# Conferência: com 1 sorteio e a mesma semente, o gerador deve repetir o forecast() do pacote.
ctx0 = np.clip(np.asarray(ds.sequences_dbz[SEL[0], :C], dtype=np.float32), 0, 60)
torch.manual_seed(SEED)
with _silencio():
  ref = gpt8.forecast(ctx0, steps=F, units='dbz', verbosity=0)
ref = _rain(np.asarray(ref.filled(0.0) if isinstance(ref, np.ma.MaskedArray) else ref))
dif_ref = float(np.max(np.abs(gerar(gpt8, ctx0, 1.0, None, 1, SEED)[0] - ref)))
CONFERE = dif_ref < 1e-3
print(f'[Célula 4.5] Conferência com o forecast() do pacote: diferença máxima {dif_ref:.2e} mm/h -> '
      + ('idêntico.' if CONFERE else 'NÃO idêntico (o padrão desta célula pode diferir da Célula 12).'))

# ------------------------------------------------------------------ 3. execução retomável
feitas = {int(os.path.basename(p)[7:12]) for p in glob.glob(os.path.join(CACHE, 'janela_*.npz'))}
pend = [i for i in SEL if int(i) not in feitas]
print(f'[Célula 4.5] {len(SEL)} janelas sorteadas da validação de 2024 | {len(SEL) - len(pend)} já prontas | '
      f'faltam {len(pend)} | {len(CONFIGS)} configurações x {len(MODELOS)} modelos | orçamento {TEMPO_MAX_MIN} min')
t_ini, tempos = time.perf_counter(), []
for n_f, i in enumerate(pend, 1):
  if (time.perf_counter() - t_ini) / 60 > TEMPO_MAX_MIN:
    print(f'[Célula 4.5] Orçamento de {TEMPO_MAX_MIN} min atingido: rode de novo para continuar.')
    break
  t_j = time.perf_counter()
  ctx = np.clip(np.asarray(ds.sequences_dbz[i, :C], dtype=np.float32), 0, 60)
  out = {}
  with _silencio():
    for nm, model in MODELOS.items():
      for k, (temp, topk, n) in enumerate(CONFIGS.values()):
        p = gerar(model, ctx, temp, topk, n, SEED + int(i) + (0 if n == 1 else 100_000))
        out[f'{nm}|{k}'] = (p.mean(axis=0) if n > 1 else p[0]).astype(np.float16)
  np.savez_compressed(os.path.join(CACHE, f'janela_{int(i):05d}.npz'), **{k.replace('|', '__'): v for k, v in out.items()})
  tempos.append(time.perf_counter() - t_j)
  if n_f == 1 or n_f % 10 == 0:
    m = float(np.mean(tempos[-20:]))
    print(f'  {len(SEL) - len(pend) + n_f}/{len(SEL)} janelas | {m:.1f} s/janela | faltam ~{(len(pend) - n_f) * m / 60:.0f} min',
          flush=True)

# ------------------------------------------------------------------ 4. montagem (só janelas prontas)
prontas = np.array(sorted(i for i in SEL if os.path.exists(os.path.join(CACHE, f'janela_{int(i):05d}.npz'))))
if len(prontas) == 0:
  raise RuntimeError('Nenhuma janela pronta ainda.')
PARCIAL = len(prontas) < len(SEL)
obs = _rain(np.asarray(ds.sequences_dbz[prontas, C:C + F]))
pers = np.repeat(_rain(np.asarray(ds.sequences_dbz[prontas, C - 1:C])), F, axis=1)
P = {}
for nm in MODELOS:
  for k, cfg in enumerate(CONFIGS):
    P[(nm, cfg)] = np.stack([np.load(os.path.join(CACHE, f'janela_{int(i):05d}.npz'))[f'{nm}__{k}'] for i in prontas]).astype(np.float32)
ts_p = np.asarray(ds.timestamps)[prontas]


# ------------------------------------------------------------------ 5. métricas
def _cont(o, p, thr):
  ob, pb = o >= thr, p >= thr
  return float(np.sum(ob & pb)), float(np.sum(~ob & pb)), float(np.sum(ob & ~pb))


def _csi(tp, fp, fn):
  return tp / (tp + fp + fn) if tp + fp + fn else np.nan


def _resumo(o, p):
  r = {}
  for thr in THRESHOLDS:
    tp, fp, fn = _cont(o, p, thr)
    r[f'CSI @ {thr}'] = _csi(tp, fp, fn)
  tp, fp, fn = _cont(o, p, 0.5)
  r['BIAS @ 0.5'] = (tp + fp) / (tp + fn) if tp + fn else np.nan
  r['MAE [mm/h]'] = float(np.mean(np.abs(p.astype(np.float64) - o)))
  r['Volume Prev/Obs'] = float(p.sum(dtype=np.float64) / o.sum(dtype=np.float64))
  return r


dias = pd.to_datetime(ts_p).normalize().values
ud, inv = np.unique(dias, return_inverse=True)
am = np.random.default_rng(SEED).integers(0, len(ud), (N_BOOT, len(ud)))


def _por_janela(p):
  ob, pb = obs >= 0.5, p >= 0.5
  return [np.bincount(inv, np.sum(a, axis=(1, 2, 3)).astype(float), len(ud)) for a in (ob & pb, ~ob & pb, ob & ~pb)]


def _boot_csi(p):
  tp, fp, fn = _por_janela(p)
  return tp[am].sum(1) / (tp[am].sum(1) + fp[am].sum(1) + fn[am].sum(1))


csi_pers_boot = _boot_csi(pers)
linhas, lead_rows = [], []
for nm in MODELOS:
  base = _boot_csi(P[(nm, PADRAO)])
  for cfg in CONFIGS:
    p = P[(nm, cfg)]
    r = {'Modelo': nm, 'Amostragem': cfg, **_resumo(obs, p)}
    b = _boot_csi(p)
    d, dp = b - base, b - csi_pers_boot
    r['ΔCSI vs padrão (IC 95%)'] = '—' if cfg == PADRAO else f'{np.percentile(d, 2.5):+.3f} a {np.percentile(d, 97.5):+.3f}'
    r['_d_lo'], r['_d_hi'] = (np.nan, np.nan) if cfg == PADRAO else (np.percentile(d, 2.5), np.percentile(d, 97.5))
    r['ΔCSI vs persistência (IC 95%)'] = f'{np.percentile(dp, 2.5):+.3f} a {np.percentile(dp, 97.5):+.3f}'
    r['_p_lo'] = np.percentile(dp, 2.5)
    linhas.append(r)
    for s in range(F):
      lead_rows.append({'Modelo': nm, 'Amostragem': cfg, 'Horizonte': STEP_MIN * (s + 1),
                        'CSI @ 0.5': _csi(*_cont(obs[:, s], p[:, s], 0.5))})
r_p = {'Modelo': 'Persistência', 'Amostragem': '—', **_resumo(obs, pers), 'ΔCSI vs padrão (IC 95%)': '—',
       'ΔCSI vs persistência (IC 95%)': '—'}
linhas.append(r_p)
for s in range(F):
  lead_rows.append({'Modelo': 'Persistência', 'Amostragem': '—', 'Horizonte': STEP_MIN * (s + 1),
                    'CSI @ 0.5': _csi(*_cont(obs[:, s], pers[:, s], 0.5))})
df = pd.DataFrame(linhas)
df_lead = pd.DataFrame(lead_rows)

print('[Célula 4.5] Previsões prontas. Calculando FSS e SAL (alguns minutos) ...', flush=True)
nomes = {f'{nm} | {cfg}': P[(nm, cfg)] for nm in MODELOS for cfg in CONFIGS}
nomes['Persistência'] = pers
df_fss = fss_over_set(obs, nomes, thr=0.5) if 'fss_over_set' in globals() else pd.DataFrame()
sel = slice(None, None, SAL_STRIDE)
df_sal = sal_over_set(obs[sel], {n: p[sel] for n, p in nomes.items()}) if 'sal_over_set' in globals() else pd.DataFrame()


def _med(dfx, nome, col):
  if not len(dfx) or col not in dfx.columns:
    return np.nan
  return float(pd.to_numeric(dfx.loc[dfx['Modelo'] == nome, col], errors='coerce').mean())


df['Chave'] = [f'{m} | {a}' if m != 'Persistência' else 'Persistência' for m, a in zip(df['Modelo'], df['Amostragem'])]
df['FSS 2 km (média dos horizontes)'] = [_med(df_fss, k, 'FSS @ 2 km') for k in df['Chave']]
df['Deslocamento SAL [km] (média das medianas)'] = [_med(df_sal, k, 'Shift_km mediana') for k in df['Chave']]
df['S do SAL (média das medianas)'] = [_med(df_sal, k, 'S mediana') for k in df['Chave']]

# ------------------------------------------------------------------ 6. decisão
_c = lambda v: f'{v:.3f}'.replace('.', ',')
textos = []
ESCOLHA = {}
for nm in MODELOS:
  d = df[(df['Modelo'] == nm) & (df['Amostragem'] != PADRAO)]
  melhores = d[d['_d_lo'] > 0].sort_values('CSI @ 0.5', ascending=False)
  base = float(df[(df['Modelo'] == nm) & (df['Amostragem'] == PADRAO)]['CSI @ 0.5'].iloc[0])
  if len(melhores):
    m = melhores.iloc[0]
    ESCOLHA[nm] = m['Amostragem']
    t = (f'{nm}: a melhor amostragem é "{m["Amostragem"]}" (CSI ≥ 0,5 mm/h {_c(m["CSI @ 0.5"])} contra {_c(base)} no '
         f'padrão; ΔCSI IC 95 % {m["ΔCSI vs padrão (IC 95%)"]}).')
    t += (' Ela supera a persistência (IC todo acima de zero).' if m['_p_lo'] > 0 else
          f' Mesmo assim não supera a persistência (CSI {_c(r_p["CSI @ 0.5"])}; ΔCSI IC 95 % {m["ΔCSI vs persistência (IC 95%)"]}).')
  else:
    ESCOLHA[nm] = PADRAO
    t = f'{nm}: nenhuma amostragem é comprovadamente melhor que o padrão (CSI {_c(base)}).'
  textos.append(t)
diag = ' '.join(textos)
if any('Média' in v for v in ESCOLHA.values()):
  diag += (' Atenção: a média de sorteios é um campo suavizado (melhora o acerto por pixel, mas não é um cenário de '
           'chuva realista; a chuva forte tende a ser atenuada). Confira o CSI ≥ 5 e 10 mm/h e o S do SAL antes de adotá-la.')

# ------------------------------------------------------------------ 7. figura
cores = dict(zip(CONFIGS, ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#8c564b', '#17becf']))
fig, axs = plt.subplots(1, 2, figsize=(16, 6.2), sharey=True)
fig.suptitle('Sensibilidade da Amostragem do GPTCast', fontsize=16, fontweight='bold', y=1.0)
hz = np.arange(1, F + 1) * STEP_MIN
for ax, nm in zip(axs, MODELOS):
  for cfg in CONFIGS:
    v = df_lead[(df_lead['Modelo'] == nm) & (df_lead['Amostragem'] == cfg)]['CSI @ 0.5'].to_numpy()
    ax.plot(hz, v, 'o-', color=cores[cfg], lw=2.4 if cfg == ESCOLHA[nm] else 1.4, label=cfg)
  vp = df_lead[df_lead['Modelo'] == 'Persistência']['CSI @ 0.5'].to_numpy()
  ax.plot(hz, vp, 's:', color='#7f7f7f', lw=2, label='Persistência')
  ax.set_title(rf'GPTCast {nm} — CSI $\geq$ 0,5 mm/h  $(\mathrm{{CSI}} = \frac{{TP}}{{TP + FP + FN}})$', fontweight='bold')
  ax.set_xlabel('Horizonte (min)')
  ax.set_xticks(hz)
  ax.grid(True, ls=':', alpha=0.6)
axs[0].set_ylabel('CSI')
axs[0].legend(fontsize=8.5, loc='upper center', bbox_to_anchor=(1.05, -0.12), ncol=4)   # abaixo dos gráficos
png = os.path.join(RESULTS_ENV_DIR, f'sensibilidade_amostragem_validacao2024_{ENV_TAG}.png')
fig.savefig(png, dpi=200, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------ 8. saída de texto e arquivos
cols = ['Modelo', 'Amostragem', 'CSI @ 0.5', 'CSI @ 5.0', 'CSI @ 10.0', 'BIAS @ 0.5', 'MAE [mm/h]', 'Volume Prev/Obs',
        'FSS 2 km (média dos horizontes)', 'Deslocamento SAL [km] (média das medianas)', 'S do SAL (média das medianas)',
        'ΔCSI vs padrão (IC 95%)', 'ΔCSI vs persistência (IC 95%)']
df[cols].round(4).to_csv(os.path.join(RESULTS_ENV_DIR, f'sensibilidade_amostragem_validacao2024_{ENV_TAG}.csv'), index=False)
df_lead.to_csv(os.path.join(RESULTS_ENV_DIR, f'sensibilidade_amostragem_por_horizonte_validacao2024_{ENV_TAG}.csv'), index=False)
print('=== Contexto (F-PluvIA Metropolitan — Célula 4.5, sensibilidade da amostragem) ===')
for ln in (
    f'Ambiente: {ENV_LABEL}  |  CUDA {torch.version.cuda}  |  Transformer em bf16, VQGAN em float32 (padrões do gptcast)',
    f'Dados: validação 2024 — {len(prontas)} de {len(SEL)} janelas sorteadas (semente {SEED}), {len(ud)} dias'
    + ('  |  RESULTADO PARCIAL: rode de novo para completar' if PARCIAL else '') + '  |  teste 2025 não usado',
    f'Gerador: igual ao predict_sequence() do pacote, com temperatura e top-k expostos; janela espacial {WS}x{WS} tokens; '
    f'conferência com o forecast(): {"idêntico" if CONFERE else "diferente"}',
    f'Média de sorteios: {N_MEMBROS} membros gerados num só lote, média em mm/h  |  previsões em {os.path.basename(CACHE)}',
):
  print('  ' + ln)
print(f'\n[Célula 4.5] Figura -> {png}\n')
with pd.option_context('display.width', 250, 'display.max_columns', 20):
  print(df[cols].round(3).set_index(['Modelo', 'Amostragem']).to_string())
print('\n=== Decisão (regra desta célula) ===')
print(textwrap.fill(diag, 110))
print(textwrap.fill('Regra: uma amostragem só substitui o padrão se o IC 95 % do ΔCSI ≥ 0,5 mm/h contra o padrão (bootstrap '
                    f'por dias, {N_BOOT} reamostragens) ficar todo acima de zero; entre as que passam, vale a de maior CSI. '
                    'A escolha é feita só em 2024; aplicá-la no teste de 2025 exige rodar a Célula 12 com essa amostragem.', 110))

CELL4_5_AMOSTRAGEM = {'escolha': ESCOLHA, 'n_janelas': int(len(prontas)), 'parcial': PARCIAL,
                      'tabela': df[cols], 'por_horizonte': df_lead, 'confere_forecast': CONFERE}
