# @title
# Célula 4.4: Teto de Reconstrução da VQGAN — codifica e decodifica pela VQGAN congelada quadros OBSERVADOS da
# validação de 2024, sem previsão nenhuma, e compara a reconstrução com o original (CSI, BIAS, MAE, PCC, FSS e SAL).
# Responde: a perda de qualidade dos dois GPTCast já acontece na tradução IMERG -> tokens -> IMERG (gargalo da
# VQGAN) ou na geração autorregressiva do Transformer? Usa só 2024: o teste de 2025 continua intocado.
# Rode depois da Célula 4 e das Células 8 e 11 (SAL e FSS). Não altera modelo, checkpoints nem previsões.
import os
import textwrap
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gptcast.utils.converters import dbz_to_rainfall

# ------------------------------------------------------------------ 1. parâmetros
N_MAX = None            # None = todas as janelas de validação de 2024
LOTE = 16               # janelas por lote na GPU (11 quadros cada)
SAL_STRIDE = 2          # SAL em 1 de cada SAL_STRIDE janelas (o SAL é a etapa mais lenta)
THRESHOLDS = (0.5, 1.0, 5.0, 10.0)
STEP_MIN = 30
ENV_TAG = globals().get('ENV_TAG', 'Ambiente_Ativo')
ENV_LABEL = globals().get('ENV_LABEL', 'Hardware Ativo')

faltam = [n for n in ('gpt8', 'md_rmf', 'RESULTS_ENV_DIR', 'CONTEXT_STEPS', 'FORECAST_STEPS') if n not in globals()]
if faltam:
  raise RuntimeError(f'Rode antes a Célula 4. Faltam: {", ".join(faltam)}')
if not torch.cuda.is_available():
  raise RuntimeError('A Célula 4.4 passa milhares de quadros pela VQGAN: ative a GPU.')
DEVICE = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# VQGAN congelada (VAEGANVQ do pacote gptcast): idêntica no Zero-Shot e no Fine-Tuned. Atenção: o GPTCast troca o
# train() da VQGAN por uma função que devolve o próprio argumento, então .eval() nela devolve False, e não o modelo.
# Por isso não se encadeia .eval() aqui; a VQGAN já fica em modo de avaliação desde a criação do GPTCast.
vq = gpt8.first_stage_model
vq.to(DEVICE)
if not hasattr(vq, 'encode'):
  raise RuntimeError('A VQGAN do GPTCast não expõe encode() nesta versão do pacote.')


def _metodos(obj):
  return sorted(n for n in dir(obj) if not n.startswith('_') and callable(getattr(obj, n, None)))


# No código do GPTCast (gptcast/models/vaegan.py), a VQGAN decodifica com decode(z) = decoder(emb_to_z(z)), que é o
# caminho usado na previsão (decode_to_img). As alternativas abaixo só entram se outra versão do pacote mudar isso.
# O caminho usado é impresso na saída.
def _escolher_decodificador():
  for nome in ('decode', 'decode_quant', 'decode_from_quant', 'decode_latent'):
    if callable(getattr(vq, nome, None)):
      return f'first_stage_model.{nome}(quant)', lambda q, idx: getattr(vq, nome)(q)
  if isinstance(getattr(vq, 'decoder', None), torch.nn.Module):
    pq = getattr(vq, 'post_quant_conv', None)
    if isinstance(pq, torch.nn.Module):
      return 'first_stage_model.decoder(post_quant_conv(quant))', lambda q, idx: vq.decoder(pq(q))
    return 'first_stage_model.decoder(quant)', lambda q, idx: vq.decoder(q)
  if callable(getattr(gpt8, 'decode_to_img', None)):
    return 'gpt8.decode_to_img(índices, forma)', lambda q, idx: gpt8.decode_to_img(idx, q.shape)
  raise RuntimeError('Não encontrei como decodificar a VQGAN. Métodos da VQGAN: '
                     f'{_metodos(vq)} | submódulos: {[n for n, _ in vq.named_children()]} | '
                     f'métodos do GPTCast com "decod": {[m for m in _metodos(gpt8) if "decod" in m.lower()]}')


CAMINHO_DECODE, _decode = _escolher_decodificador()

ds = md_rmf.splits['val']
S = ds.sequences_dbz if N_MAX is None else ds.sequences_dbz[:N_MAX]
N, T, H, W = S.shape
C, F = int(CONTEXT_STEPS), int(FORECAST_STEPS)
ts = np.asarray(ds.timestamps)[:N]


def _rain(dbz):
  return np.clip(dbz_to_rainfall(np.clip(np.asarray(dbz, dtype=np.float64), 0.0, 60.0)), 0.0, 205.0).astype(np.float32)


# ------------------------------------------------------------------ 2. reconstrução na GPU
# Mesma normalização do treino (Célula 3): dBZ [0, 60] -> [-1, 1]. Dois modos:
#  - quadro isolado: cada quadro passa sozinho pela VQGAN (fidelidade pura da VQGAN);
#  - empilhado: os 11 quadros da janela empilhados na vertical, como na tokenização da previsão.
@torch.inference_mode()
def reconstruir(lote_dbz, empilhado):
  x = torch.from_numpy(np.clip(lote_dbz, 0, 60) / 30.0 - 1.0).float().to(DEVICE, non_blocking=True)   # (B, T, H, W)
  b, t, h, w = x.shape
  x = x.reshape(b, 1, t * h, w) if empilhado else x.reshape(b * t, 1, h, w)
  quant, _, info = vq.encode(x)
  idx = info.view(quant.shape[0], -1) if torch.is_tensor(info) else (info[2].view(quant.shape[0], -1)
                                                                       if isinstance(info, (tuple, list)) else None)
  rec = _decode(quant, idx)
  rec = (rec[0] if isinstance(rec, (tuple, list)) else rec).float().clamp(-1, 1)
  return ((rec.reshape(b, t, h, w) + 1.0) * 30.0).cpu().numpy()


obs_t = _rain(S[:, C:C + F])                                     # quadros-alvo observados (N, 4, H, W), mm/h
rec_iso = np.empty_like(obs_t)
rec_stk = np.empty_like(obs_t)
torch.cuda.reset_peak_memory_stats(0)
torch.cuda.synchronize()
t0 = time.perf_counter()
for a in range(0, N, LOTE):
  bloco = np.asarray(S[a:a + LOTE], dtype=np.float32)
  rec_iso[a:a + LOTE] = _rain(reconstruir(bloco, False)[:, C:C + F])
  rec_stk[a:a + LOTE] = _rain(reconstruir(bloco, True)[:, C:C + F])
torch.cuda.synchronize()
t_rec = time.perf_counter() - t0
pers = np.repeat(_rain(S[:, C - 1:C]), F, axis=1)               # persistência do último quadro de contexto

METODOS = {'Reconstrução VQGAN (quadro isolado)': rec_iso, 'Reconstrução VQGAN (empilhada)': rec_stk,
           'Persistência (referência)': pers}


# ------------------------------------------------------------------ 3. métricas por pixel
def _linha(o, p, nome, rotulo):
  o64, p64 = o.astype(np.float64), p.astype(np.float64)
  r = {'Método': nome, 'Quadros': rotulo, 'MAE [mm/h]': float(np.mean(np.abs(p64 - o64))),
       'PCC': float(np.corrcoef(o64.ravel(), p64.ravel())[0, 1]), 'Volume Prev/Obs': float(p64.sum() / o64.sum())}
  for thr in THRESHOLDS:
    ob, pb = o >= thr, p >= thr
    tp, fp, fn = float(np.sum(ob & pb)), float(np.sum(~ob & pb)), float(np.sum(ob & ~pb))
    r[f'CSI @ {thr}'] = tp / (tp + fp + fn) if tp + fp + fn else np.nan
    r[f'BIAS @ {thr}'] = (tp + fp) / (tp + fn) if tp + fn else np.nan
  return r


linhas = []
for nome, P in METODOS.items():
  if nome.startswith('Persist'):
    for s in range(F):
      linhas.append(_linha(obs_t[:, s], P[:, s], nome, f'+{STEP_MIN * (s + 1)} min'))
  else:
    linhas.append(_linha(obs_t, P, nome, 'todos os alvos'))
df = pd.DataFrame(linhas)

csi_iso = float(df.loc[df['Método'].str.contains('isolado'), 'CSI @ 0.5'].iloc[0])
csi_stk = float(df.loc[df['Método'].str.contains('empilhada'), 'CSI @ 0.5'].iloc[0])
pers_csi = df.loc[df['Método'].str.startswith('Persist'), 'CSI @ 0.5'].to_numpy(float)
csi_rec = max(csi_iso, csi_stk)

# ------------------------------------------------------------------ 4. FSS e SAL (Células 11 e 8)
print(f'[Célula 4.4] Reconstrução de {N} janelas ({N * F} quadros-alvo, dois modos) em {t_rec:.0f} s. '
      'Calculando FSS e SAL (alguns minutos) ...', flush=True)
df_fss = fss_over_set(obs_t, METODOS, thr=0.5) if 'fss_over_set' in globals() else pd.DataFrame()
sel = slice(None, None, SAL_STRIDE)
df_sal = (sal_over_set(obs_t[sel], {n: p[sel] for n, p in METODOS.items()})
          if 'sal_over_set' in globals() else pd.DataFrame())


def _sal(nome, col):
  if not len(df_sal) or col not in df_sal.columns:
    return np.nan
  return float(pd.to_numeric(df_sal.loc[df_sal['Modelo'] == nome, col], errors='coerce').median())


S_rec = np.nanmin([_sal('Reconstrução VQGAN (quadro isolado)', 'S mediana'),
                   _sal('Reconstrução VQGAN (empilhada)', 'S mediana')])

# ------------------------------------------------------------------ 5. diagnóstico
_c = lambda v: f'{v:.3f}'.replace('.', ',')
if csi_rec < pers_csi[0]:
  diag = (f'A VQGAN é um gargalo: reconstruindo quadros observados, sem prever nada, o CSI ≥ 0,5 mm/h fica em '
          f'{_c(csi_rec)}, abaixo da persistência de +30 min ({_c(pers_csi[0])}). Nenhum Transformer, ajustado ou não, '
          'pode superar esse teto; isso explica boa parte do desempenho dos dois GPTCast no teste de 2025.')
elif csi_rec < pers_csi[0] + 0.05:
  diag = (f'A VQGAN limita, mas não explica tudo sozinha: a reconstrução (CSI {_c(csi_rec)}) fica no nível da '
          f'persistência de +30 min ({_c(pers_csi[0])}). Sobra pouca margem para o Transformer superar a extrapolação; '
          'parte da perda dos GPTCast vem também da geração autorregressiva.')
else:
  diag = (f'A VQGAN reconstrói bem: CSI {_c(csi_rec)}, acima da persistência de +30 min ({_c(pers_csi[0])}). O '
          'gargalo dos GPTCast está na geração autorregressiva (sorteio de tokens e deriva ao longo do horizonte), '
          'não na tradução IMERG -> tokens.')
if np.isfinite(S_rec) and S_rec < -0.3:
  diag += (f' O componente S do SAL da reconstrução ({_c(S_rec)}) é fortemente negativo, como nos dois GPTCast: '
           'a própria VQGAN deixa os objetos de chuva menores e mais pontudos.')
elif np.isfinite(S_rec):
  diag += (f' O componente S do SAL da reconstrução ({_c(S_rec)}) é pequeno: a deformação de estrutura vista nos '
           'GPTCast não vem da VQGAN, e sim da geração.')
if abs(csi_iso - csi_stk) > 0.02:
  diag += (f' O empilhamento muda a reconstrução (isolado {_c(csi_iso)} × empilhado {_c(csi_stk)}): as convoluções '
           'misturam informação entre quadros vizinhos na pilha.')

# ------------------------------------------------------------------ 6. figura
fig = plt.figure(figsize=(17, 6.2))
gs = fig.add_gridspec(1, 4, width_ratios=[1.6, 1, 1, 1], wspace=0.25)
fig.suptitle('Teto de Reconstrução da VQGAN', fontsize=16, fontweight='bold', y=1.0)

ax = fig.add_subplot(gs[0, 0])
hz = np.arange(1, F + 1) * STEP_MIN
ax.plot(hz, pers_csi, 's:', color='#7f7f7f', lw=2, label='Persistência (validação 2024)')
ax.axhline(csi_iso, color='#2ca02c', lw=2.2, label=f'Reconstrução isolada: {csi_iso:.3f}')
ax.axhline(csi_stk, color='#9467bd', lw=2.2, ls='--', label=f'Reconstrução empilhada: {csi_stk:.3f}')
r12 = globals().get('CELL12_FULL_2025_RESULTS')
if isinstance(r12, dict) and 'ft_rain' in r12:
  for k, nome, cor, mk in (('ft_rain', 'Fine-Tuned', '#d62728', 'o'), ('zs_rain', 'Zero-Shot', '#ff7f0e', 'v')):
    v = [_linha(r12['obs_rain'][:, s], r12[k][:, s], nome, '')['CSI @ 0.5'] for s in range(F)]
    ax.plot(hz, v, marker=mk, color=cor, lw=1.2, alpha=0.55, label=f'GPTCast {nome} (teste 2025, referência)')
ax.set_title(r'CSI $\geq$ 0,5 mm/h  $(\mathrm{CSI} = \frac{TP}{TP + FP + FN})$', fontweight='bold')
ax.set_xlabel('Horizonte (min)')
ax.set_ylabel('CSI')
ax.set_xticks(hz)
ax.grid(True, ls=':', alpha=0.6)
ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2)   # abaixo do gráfico: não cobre as linhas

i_ex = int(np.argmax((obs_t[:, 0] >= 0.5).sum(axis=(1, 2))))   # janela com maior área de chuva no 1º alvo
vmax = float(np.percentile(obs_t[i_ex, 0][obs_t[i_ex, 0] > 0], 99)) if (obs_t[i_ex, 0] > 0).any() else 10.0
for j, (img, tit) in enumerate(((obs_t[i_ex, 0], 'Observado (IMERG)'), (rec_iso[i_ex, 0], 'Reconstrução isolada'),
                                (rec_stk[i_ex, 0], 'Reconstrução empilhada'))):
  a = fig.add_subplot(gs[0, j + 1])
  im = a.imshow(img, cmap='turbo', vmin=0, vmax=vmax)
  a.set_title(tit, fontweight='bold', fontsize=11)
  a.set_xticks([]); a.set_yticks([])
fig.colorbar(im, ax=fig.axes[1:], shrink=0.75, label='mm/h')

png = os.path.join(RESULTS_ENV_DIR, f'teto_reconstrucao_vqgan_validacao2024_{ENV_TAG}.png')
fig.savefig(png, dpi=200, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------ 7. saída de texto e arquivos
df.round(4).to_csv(os.path.join(RESULTS_ENV_DIR, f'teto_vqgan_metricas_validacao2024_{ENV_TAG}.csv'), index=False)
if len(df_fss):
  df_fss.to_csv(os.path.join(RESULTS_ENV_DIR, f'teto_vqgan_fss_validacao2024_{ENV_TAG}.csv'), index=False)
if len(df_sal):
  df_sal.to_csv(os.path.join(RESULTS_ENV_DIR, f'teto_vqgan_sal_validacao2024_{ENV_TAG}.csv'), index=False)

print('=== Contexto (F-PluvIA Metropolitan — Célula 4.4, teto de reconstrução da VQGAN) ===')
for ln in (
    f'Ambiente: {ENV_LABEL}  |  CUDA {torch.version.cuda}  |  VQGAN congelada em float32 + TF32  |  '
    f'pico de VRAM {torch.cuda.max_memory_allocated(0) / 2**30:.1f} GB',
    f'Dados: validação 2024 — {N} janelas, {N * F} quadros-alvo (posições {C + 1} a {C + F} da janela), '
    f'{len(np.unique(pd.to_datetime(ts).normalize()))} dias  |  teste 2025 não usado',
    'Modos: quadro isolado (cada quadro sozinho) e empilhado (11 quadros na vertical, como na previsão)',
    f'Decodificação: {CAMINHO_DECODE}',
    f'Exemplo da figura: janela com t final {str(ts[i_ex])[:16]} (maior área de chuva no 1º quadro-alvo)',
):
  print('  ' + ln)
print(f'\n[Célula 4.4] Figura -> {png}\n')
cols = ['Método', 'Quadros', 'CSI @ 0.5', 'CSI @ 1.0', 'CSI @ 5.0', 'CSI @ 10.0', 'BIAS @ 0.5', 'MAE [mm/h]', 'PCC',
        'Volume Prev/Obs']
print(df[cols].round(4).set_index(['Método', 'Quadros']).to_string())
if len(df_fss):
  cf = [c for c in df_fss.columns if c.startswith('FSS @')][:6] + [c for c in df_fss.columns if 'Escala' in c]
  print('\n--- FSS (≥ 0,5 mm/h) ---')
  print(df_fss.set_index(['Modelo', 'Lead Time'])[cf].round(4).to_string())
if len(df_sal):
  cs = [c for c in ('S mediana', 'A mediana', 'L mediana', 'Shift_km mediana', 'Razao_Volume mediana') if c in df_sal.columns]
  print(f'\n--- SAL (mediana; 1 de cada {SAL_STRIDE} janelas) ---')
  print(df_sal.set_index(['Modelo', 'Lead Time'])[cs].to_string())
print('\nNa reconstrução, "Lead Time" é só a posição do quadro-alvo: a VQGAN não prevê nada, apenas reconstrói o quadro observado.')
print('\n=== Diagnóstico automático (regras desta célula) ===')
print(textwrap.fill(diag, 110))
print(textwrap.fill('Regras: compara o melhor CSI ≥ 0,5 mm/h da reconstrução com a persistência de +30 min na mesma '
                    'validação de 2024. Abaixo dela = VQGAN é gargalo; até 0,05 acima = limita, mas não explica tudo; '
                    'mais que isso = gargalo na geração. S do SAL da reconstrução abaixo de −0,3 = a VQGAN deforma a '
                    'estrutura da chuva. As curvas do GPTCast na figura são do teste de 2025 e servem só de referência '
                    'visual (anos diferentes).', 110))

CELL4_4_TETO_VQGAN = {'csi05_reconstrucao_isolada': csi_iso, 'csi05_reconstrucao_empilhada': csi_stk,
                      'csi05_persistencia_por_horizonte': pers_csi.tolist(), 'sal_S_reconstrucao': S_rec,
                      'n_janelas': N, 'diagnostico': diag}
