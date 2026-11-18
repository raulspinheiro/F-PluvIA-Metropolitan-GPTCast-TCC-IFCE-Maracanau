# @title
# Célula 4.3 (Opcional - Fine-Tuning Suave - Adds Another Dimension of Search): novo fine-tuning com taxa de aprendizado 1e-5 e validação a cada
# 1/4 de época, comparado ao treino atual da Célula 4 nas mesmas janelas de 2024 (IC 95 % por bootstrap de dias).
# Decide qual dos dois treinamentos fica. Não altera o modelo em uso, a não ser que PROMOVER_SE_MELHOR = True e o
# alternativo seja comprovadamente melhor.
#
# Requisito de Execução: rodar depois da Célula 4;
#
import copy
import datetime
import glob
import json
import os
import re
import shutil
import textwrap
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ------------------------------------------------------------------ 1. parâmetros
LR_ALT = 1e-5                 # taxa de aprendizado efetiva do treino alternativo (Célula 4: 5e-5)
VAL_CHECK = 0.25              # validação a cada 1/4 de época
MAX_EPOCAS_ALT = 5
PACIENCIA_VAL = 4             # em validações: 4 x 1/4 = 1 época sem melhora
FORCAR_TREINO_ALT = False     # True = treina de novo mesmo havendo checkpoint alternativo com estes parâmetros
PROMOVER_SE_MELHOR = False    # True = se o alternativo for comprovadamente melhor, ele passa a ser o modelo do projeto
N_BOOT = 2000
TOKENS_QUADRO = 64

ENV_TAG = globals().get('ENV_TAG', 'Ambiente_Ativo')
ENV_LABEL = globals().get('ENV_LABEL', 'Hardware Ativo')
SEED = int(globals().get('SEED', 42))

faltam = [n for n in ('gpt8', 'gpt8_zeroshot', 'md_rmf', 'GPTCast', 'CHECKPOINTS_ENV_DIR', 'RESULTS_ENV_DIR',
                      'training_record_json') if n not in globals()]
if faltam:
  raise RuntimeError(f'Rode antes a Célula 4. Faltam: {", ".join(faltam)}')
if not torch.cuda.is_available():
  raise RuntimeError('A Célula 4.3 treina um modelo: rode com GPU.')
if getattr(GPTCast.encode_to_z, '__name__', '') != '_spatiotemporal_encode_to_z':
  raise RuntimeError('A tokenização empilhada da Célula 4 não está ativa nesta sessão: rode a Célula 4 antes.')

device = 'cuda'
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

rec_atual = json.load(open(training_record_json, encoding='utf-8'))
if rec_atual.get('tokenizacao_treino') != 'empilhada':
  raise RuntimeError('O treino atual não está registrado com tokenização empilhada: rode a Célula 4 antes.')
BATCH = int(rec_atual.get('batch_size_treino') or globals().get('BATCH_SIZE_AUTO', 16))
ACUM = int(rec_atual.get('acumulo_gradiente') or globals().get('ACCUM_STEPS_AUTO', 1))
ckpt_atual = os.path.join(CHECKPOINTS_ENV_DIR, rec_atual['checkpoint'])

# Pasta própria: a busca de checkpoints da Célula 4 não a enxerga, então nada muda no projeto por acidente.
ALT_DIR = os.path.join(CHECKPOINTS_ENV_DIR, 'alternativos')
os.makedirs(ALT_DIR, exist_ok=True)
TAG_ALT = f'lr{LR_ALT:.0e}_val{VAL_CHECK:g}'.replace('e-0', 'e-')
registro_alt_json = os.path.join(ALT_DIR, f'registro_treino_alternativo_{TAG_ALT}_{ENV_TAG}.json')


def _carregar_pesos(model, path):
  state = torch.load(path, map_location=device)['state_dict']
  res = model.load_state_dict(state, strict=False)
  if any(k.startswith('transformer.') for k in res.missing_keys):
    raise RuntimeError(f'{os.path.basename(path)} não contém os pesos do Transformer.')
  return model.to(device).eval()


def _sync():
  torch.cuda.synchronize()
  return time.perf_counter()


# ------------------------------------------------------------------ 2. treino alternativo (ou reaproveitamento)
rec_alt = None
if not FORCAR_TREINO_ALT and os.path.isfile(registro_alt_json):
  _r = json.load(open(registro_alt_json, encoding='utf-8'))
  if os.path.isfile(os.path.join(CHECKPOINTS_ENV_DIR, _r.get('checkpoint', ''))):
    rec_alt = _r

if rec_alt is not None:
  ckpt_alt = os.path.join(CHECKPOINTS_ENV_DIR, rec_alt['checkpoint'])
  gpt8_alt = _carregar_pesos(copy.deepcopy(gpt8_zeroshot), ckpt_alt)
  print(f'[Célula 4.3] Treino alternativo já existe ({rec_alt["data_treino_utc"]}): {rec_alt["checkpoint"]}. '
        'Reaproveitado (FORCAR_TREINO_ALT = True treina de novo).')
else:
  import lightning.pytorch as pl_modern
  import pytorch_lightning as pl_legacy
  pl_active = pl_modern if issubclass(GPTCast, pl_modern.LightningModule) else pl_legacy
  pl_active.seed_everything(SEED, workers=True)
  precision_mode = 'bf16-mixed' if torch.cuda.is_bf16_supported() else '16-mixed'

  # Mesmo ponto de partida da Célula 4: pesos do Zenodo, VQGAN congelada, Transformer treinável.
  gpt8_alt = copy.deepcopy(gpt8_zeroshot).to(device)
  gpt8_alt.train()
  gpt8_alt.first_stage_model.eval()
  for p in gpt8_alt.first_stage_model.parameters():
    p.requires_grad = False
  for p in gpt8_alt.transformer.parameters():
    p.requires_grad = True
  gpt8_alt.base_learning_rate = LR_ALT / (BATCH * ACUM)
  gpt8_alt.learning_rate = LR_ALT

  run_tag = datetime.datetime.utcnow().strftime('%Y%m%dT%H%MZ')
  ckpt_cb = pl_active.callbacks.ModelCheckpoint(
      dirpath=ALT_DIR,
      filename=f'gptcast8_rmf_2km_{ENV_TAG}_{run_tag}_{TAG_ALT}_best-{{epoch:02d}}-{{step:05d}}-{{val/loss:.4f}}',
      auto_insert_metric_name=False, monitor='val/loss', mode='min', save_top_k=1, save_last=False,
  )
  stop_cb = pl_active.callbacks.EarlyStopping(monitor='val/loss', patience=PACIENCIA_VAL, mode='min', verbose=False)
  logger = pl_active.loggers.CSVLogger(save_dir=RESULTS_ENV_DIR, name=f'curva_treino_alternativo_{ENV_TAG}',
                                       version=f'{run_tag}_{TAG_ALT}')
  trainer = pl_active.Trainer(
      accelerator='gpu', devices=1, max_epochs=MAX_EPOCAS_ALT, precision=precision_mode,
      accumulate_grad_batches=ACUM, val_check_interval=VAL_CHECK, callbacks=[ckpt_cb, stop_cb],
      logger=logger, log_every_n_steps=5, deterministic=False, enable_model_summary=False,
  )
  print(f'[Célula 4.3] Treino alternativo: lr {LR_ALT:g} (atual {rec_atual.get("learning_rate_efetiva")}), '
        f'validação a cada {VAL_CHECK:g} de época, até {MAX_EPOCAS_ALT} épocas, paciência {PACIENCIA_VAL} validações, '
        f'lote {BATCH} x acúmulo {ACUM}, {precision_mode}, semente {SEED}.')
  torch.cuda.reset_peak_memory_stats(0)
  t0 = _sync()
  trainer.fit(gpt8_alt, datamodule=md_rmf)
  t_fit = _sync() - t0
  ckpt_alt = ckpt_cb.best_model_path
  if not ckpt_alt or not os.path.isfile(ckpt_alt):
    raise RuntimeError('O treino alternativo terminou sem checkpoint de melhor val/loss.')
  gpt8_alt = _carregar_pesos(gpt8_alt, ckpt_alt)
  m = re.search(r'best-(\d+)-(\d+)-', os.path.basename(ckpt_alt))
  rec_alt = {
      'data_treino_utc': run_tag, 'checkpoint': os.path.relpath(ckpt_alt, CHECKPOINTS_ENV_DIR),
      'epocas_executadas': int(trainer.current_epoch), 'melhor_epoca': int(m.group(1)) if m else None,
      'melhor_passo': int(m.group(2)) if m else None,
      'melhor_val_loss_2024': round(float(ckpt_cb.best_model_score.cpu().item()), 4),
      'finetuning_tempo_total_s': round(t_fit, 2), 'finetuning_tempo_total_min': round(t_fit / 60, 2),
      'finetuning_tempo_medio_por_epoca_s': round(t_fit / max(int(trainer.current_epoch), 1), 2),
      'precisao_numerica': precision_mode, 'batch_size_treino': BATCH, 'acumulo_gradiente': ACUM,
      'learning_rate_efetiva': LR_ALT, 'val_check_interval': VAL_CHECK, 'paciencia_validacoes': PACIENCIA_VAL,
      'semente_treino': SEED, 'corte_validacao': rec_atual.get('corte_validacao'), 'tokenizacao_treino': 'empilhada',
      'gpu': torch.cuda.get_device_name(0), 'pico_vram_treino_gb': round(torch.cuda.max_memory_allocated(0) / 2**30, 2),
      'curva_por_epoca_csv': os.path.join(logger.log_dir, 'metrics.csv'),
  }
  with open(registro_alt_json, 'w') as f:
    json.dump(rec_alt, f, indent=2, ensure_ascii=False)
  print(f'[Célula 4.3] Treino alternativo concluído em {t_fit / 60:.1f} min | melhor val/loss '
        f'{rec_alt["melhor_val_loss_2024"]:.4f} (época {rec_alt["melhor_epoca"]}, passo {rec_alt["melhor_passo"]}).')

gpt8 = gpt8.to(device).eval()
gpt8_zeroshot = gpt8_zeroshot.to(device).eval()


# ------------------------------------------------------------------ 3. mesma medição da Célula 4.1, três modelos
ds_val = md_rmf.splits['val']
loader_val = DataLoader(ds_val, batch_size=64, shuffle=False, num_workers=min(8, os.cpu_count() or 2),
                        pin_memory=True, prefetch_factor=4)


@torch.inference_mode()
def perdas_por_janela(model):
  model.eval()
  out = []
  for batch in loader_val:
    batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
    x, c = model.get_xc(batch)
    with torch.autocast('cuda', dtype=torch.bfloat16):
      logits, target = model(x, c)
    ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), target.reshape(-1), reduction='none')
    out.append(ce.view(target.shape))
  return torch.cat(out).cpu().numpy()


t0 = _sync()
ce = {'zs': perdas_por_janela(gpt8_zeroshot), 'atual': perdas_por_janela(gpt8), 'alt': perdas_por_janela(gpt8_alt)}
t_med = _sync() - t0
N = ce['atual'].shape[0]
win = {k: v.mean(1) for k, v in ce.items()}
L = {k: float(v.mean()) for k, v in win.items()}

ts = np.asarray(ds_val.timestamps)[:N]
ud, inv = np.unique(pd.to_datetime(ts).normalize().values, return_inverse=True)
cont_d = np.bincount(inv, minlength=len(ud))
am = np.random.default_rng(SEED).integers(0, len(ud), (N_BOOT, len(ud)))


def _ic(delta):
  soma = np.bincount(inv, delta, len(ud))
  boot = soma[am].sum(1) / cont_d[am].sum(1)
  return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


delta = win['alt'] - win['atual']                       # < 0: alternativo melhor
D, ic = float(delta.mean()), _ic(delta)
ic_alt_zs = _ic(win['alt'] - win['zs'])
ganho = 100 * (1 - np.exp(D))
ganho_ic = (100 * (1 - np.exp(ic[1])), 100 * (1 - np.exp(ic[0])))
frac_alt = float(np.mean(delta < 0))

ctx = int(globals().get('CONTEXT_STEPS', 7))
n_win = int(ds_val.sequences_dbz.shape[1])
n_ctx = ce['atual'].shape[1] // TOKENS_QUADRO
ini = {'start': 0, 'end': n_win - n_ctx}.get(getattr(ds_val, 'crop', 'end'), 0)
quadros = np.arange(ini + 1, ini + n_ctx + 1)
prev = quadros > ctx
fr = {k: v.reshape(N, -1, TOKENS_QUADRO).mean(2) for k, v in ce.items()}
D_prev = float((fr['alt'][:, prev] - fr['atual'][:, prev]).mean())

# ------------------------------------------------------------------ 4. decisão (IC 95 % como único critério)
_n = lambda v: f'{v:+.4f}'.replace('.', ',')
_p = lambda v: f'{v:.1f}'.replace('.', ',') + ' %'
txt_d = f'Δ = {_n(D)} nats por token, IC 95 % de {_n(ic[0])} a {_n(ic[1])}'
alt_melhor = ic[1] < 0
if alt_melhor:
  diag = (f'O treino alternativo (lr {LR_ALT:g}) é melhor que o atual ({txt_d}): perplexidade {_p(ganho)} menor '
          f'(IC 95 % de {_p(ganho_ic[0])} a {_p(ganho_ic[1])}). Vale promovê-lo a modelo do projeto.')
elif ic[0] > 0:
  diag = (f'O treino atual (lr {rec_atual.get("learning_rate_efetiva")}) é melhor que o alternativo ({txt_d}). '
          'Mantenha o treino atual; nada muda no projeto.')
else:
  diag = (f'Não há diferença comprovada entre os dois treinos ({txt_d}, que inclui zero). Mantenha o treino atual, '
          'que já foi avaliado nas células seguintes; nada muda no projeto.')

promovido = False
if alt_melhor and PROMOVER_SE_MELHOR:
  bkp = training_record_json.replace('.json', f'_antes_celula43_{datetime.datetime.utcnow():%Y%m%dT%H%MZ}.json')
  shutil.copy2(training_record_json, bkp)
  with open(training_record_json, 'w') as f:
    json.dump(dict(rec_alt, promovido_pela_celula_4_3=True, registro_anterior=os.path.basename(bkp)), f,
              indent=2, ensure_ascii=False)
  gpt8, best_ckpt_path, promovido = gpt8_alt, ckpt_alt, True
elif not alt_melhor:
  del gpt8_alt
  torch.cuda.empty_cache()

# ------------------------------------------------------------------ 5. figura
curva = None
try:
  mcsv = pd.read_csv(rec_alt['curva_por_epoca_csv'])
  col_v = next(c for c in ('val/loss_epoch', 'val/loss') if c in mcsv.columns)
  v = mcsv.dropna(subset=[col_v])[['epoch', 'step', col_v]].drop_duplicates('step')
  v['x'] = v['epoch'] + (v.groupby('epoch').cumcount() + 1) * VAL_CHECK
  curva = v
  col_t = next((c for c in ('train/loss_step', 'train/loss') if c in mcsv.columns), None)
  passos_epoca = max(1, len(md_rmf.train_dataloader()) // ACUM)
  tr = mcsv.dropna(subset=[col_t])[['step', col_t]] if col_t else None
except Exception as e:
  print(f'[Aviso Célula 4.3] Curva do treino alternativo indisponível ({e}).')
  tr = None

fig = plt.figure(figsize=(16, 6.4))
gs = fig.add_gridspec(1, 2, wspace=0.2)
fig.suptitle('Treino Alternativo × Treino Atual', fontsize=16, fontweight='bold', y=0.99)

ax1 = fig.add_subplot(gs[0, 0])
if tr is not None and len(tr):
  y = tr[tr.columns[1]].rolling(25, min_periods=1).mean()
  ax1.plot(tr['step'] / passos_epoca, y, color='#1f77b4', lw=1.4, alpha=0.8, label='Treino alternativo (média móvel)')
if curva is not None:
  ax1.plot(curva['x'], curva[curva.columns[2]], 's-', color='#9467bd', lw=2,
           label=f'Validação alternativo (lr {LR_ALT:g}, a cada {VAL_CHECK:g} época)')
  i = curva[curva.columns[2]].idxmin()
  ax1.scatter([curva.loc[i, 'x']], [curva.loc[i, curva.columns[2]]], s=160, facecolors='none',
              edgecolors='#2ca02c', lw=2.5, zorder=5, label='Melhor validação do alternativo')
ax1.axhline(L['atual'], color='#d62728', ls='--', lw=1.8, label=f'Treino atual (lr {rec_atual.get("learning_rate_efetiva")}): {L["atual"]:.4f}')
ax1.axhline(L['zs'], color='#ff7f0e', ls=':', lw=1.8, label=f'Zero-Shot: {L["zs"]:.4f}')
ax1.set_title(r'Perda ao longo do treino  $(\mathcal{L} = -\frac{1}{N}\,\Sigma_{i=1}^{N}\, \log p_\theta(z_i \mid z_{<i}))$',
              fontweight='bold')
ax1.set_xlabel('Época (fração)')
ax1.set_ylabel(r'$\mathcal{L}$ (entropia cruzada, nats por token)')
ax1.grid(True, ls=':', alpha=0.6)
ax1.legend(fontsize=8.5)

ax2 = fig.add_subplot(gs[0, 1])
lim = np.percentile(np.abs(delta), 99.5)
ax2.hist(np.clip(delta, -lim, lim), bins=60, color='#7f7f7f', alpha=0.75)
ax2.axvline(0, color='k', lw=1)
ax2.axvline(D, color='#9467bd', lw=2.2, label=f'Média Δ = {D:+.4f}')
ax2.axvspan(ic[0], ic[1], color='#9467bd', alpha=0.2, label=f'IC 95 % (bootstrap por {len(ud)} dias)')
ax2.set_title(r'Diferença por janela  $(\Delta_j = \mathcal{L}_{\mathrm{alt},j} - \mathcal{L}_{\mathrm{atual},j})$',
              fontweight='bold')
ax2.set_xlabel(r'$\Delta_j$ (negativo = alternativo melhor)')
ax2.set_ylabel('Janelas de 2024')
ax2.grid(True, axis='y', ls=':', alpha=0.6)
ax2.legend(fontsize=8.5)

png = os.path.join(RESULTS_ENV_DIR, f'treino_alternativo_{TAG_ALT}_vs_atual_{ENV_TAG}.png')
fig.savefig(png, dpi=200, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------ 6. saída de texto
resumo = pd.DataFrame([
    {'Modelo': nome, 'lr': lr, 'val/loss (nats/token)': round(L[k], 4), 'Perplexidade (exp L)': round(float(np.exp(L[k])), 1),
     'val/loss quadros 8–11': round(float(fr[k][:, prev].mean()), 4)}
    for k, nome, lr in (('zs', 'Zero-Shot', '—'), ('atual', 'Treino atual', rec_atual.get('learning_rate_efetiva')),
                        ('alt', 'Treino alternativo', LR_ALT))
]).set_index('Modelo')
resumo.assign(Delta_alt_menos_atual=round(D, 4), IC95_inf=round(ic[0], 4), IC95_sup=round(ic[1], 4),
              Janelas=N, Dias=len(ud), Promovido=promovido).to_csv(
    os.path.join(RESULTS_ENV_DIR, f'treino_alternativo_{TAG_ALT}_resumo_{ENV_TAG}.csv'))
pd.DataFrame({'t0_fim_janela': ts, 'val_loss_zero_shot': win['zs'], 'val_loss_atual': win['atual'],
              'val_loss_alternativo': win['alt'], 'delta_alt_menos_atual': delta}).to_csv(
    os.path.join(RESULTS_ENV_DIR, f'treino_alternativo_{TAG_ALT}_por_janela_{ENV_TAG}.csv'), index=False)

print('=== Contexto (F-PluvIA Metropolitan — Célula 4.3, treino alternativo) ===')
for ln in (
    f'Ambiente: {ENV_LABEL}  |  CUDA {torch.version.cuda}  |  medição em bf16 + TF32, lote 64, {t_med:.0f} s para os 3 modelos',
    f'Treino atual: {os.path.basename(ckpt_atual)}  |  lr {rec_atual.get("learning_rate_efetiva")}  |  validação por época  |  '
    f'{rec_atual.get("epocas_executadas")} épocas, melhor época {rec_atual.get("melhor_epoca")}',
    f'Treino alternativo: {os.path.basename(ckpt_alt)}  |  lr {LR_ALT:g}  |  validação a cada {VAL_CHECK:g} época  |  '
    f'{rec_alt.get("epocas_executadas")} épocas em {rec_alt.get("finetuning_tempo_total_min")} min, melhor na época '
    f'{rec_alt.get("melhor_epoca")} (passo {rec_alt.get("melhor_passo")})',
    f'Dados: validação 2024 — {N} janelas em {len(ud)} dias; mesmo ponto de partida (Zenodo), lote, semente e tokenização empilhada',
):
  print('  ' + ln)
print(f'\n[Célula 4.3] Figura -> {png}\n')
print(resumo.to_string())
print(f'\nΔ = val/loss(alternativo) − val/loss(atual) = {D:+.4f} | IC 95 %: {ic[0]:+.4f} a {ic[1]:+.4f} '
      f'(bootstrap de {N_BOOT} reamostragens de {len(ud)} dias)')
print(f'Ganho em perplexidade do alternativo sobre o atual: {ganho:.1f} % (IC 95 %: {ganho_ic[0]:.1f} a {ganho_ic[1]:.1f} %)')
print(f'Alternativo melhor em {100 * frac_alt:.1f} % das janelas | Δ nos quadros previstos (8 a 11): {D_prev:+.4f}')
print(f'Alternativo × Zero-Shot: {L["alt"] - L["zs"]:+.4f} (IC 95 %: {ic_alt_zs[0]:+.4f} a {ic_alt_zs[1]:+.4f})')
print('\n=== Decisão (regra desta célula) ===')
print(textwrap.fill(diag, 110))
if alt_melhor and not promovido:
  print(textwrap.fill('Para promover: mude PROMOVER_SE_MELHOR para True e rode esta célula de novo (o checkpoint é '
                      'reaproveitado, sem novo treino).', 110))
if promovido:
  print(textwrap.fill(f'PROMOVIDO: registro_treino_{ENV_TAG}.json agora aponta para {rec_alt["checkpoint"]} (cópia do '
                      'registro anterior guardada ao lado). gpt8 e best_ckpt_path já são o alternativo nesta sessão. '
                      'Rode de novo a Célula Activates Checkpoint, a 4.1, a 4.2 e as Células 5 a 13.1; as Células 12 e '
                      '13.1 refazem as previsões, porque a impressão digital do modelo mudou.', 110))
print(textwrap.fill('Regra: o IC 95 % da diferença (bootstrap por dias) é o único critério. Observação: os dois treinos '
                    'foram escolhidos pela mesma validação de 2024, então a comparação final entre eles deve ser '
                    'confirmada no teste de 2025 (Células 9 a 13.1).', 110))

CELL4_3_TREINO_ALT = {'lr_alternativo': LR_ALT, 'val_check_interval': VAL_CHECK, 'val_loss': L, 'delta_alt_atual': D,
                      'ic95': ic, 'ganho_perplexidade_pct': float(ganho), 'promovido': promovido,
                      'checkpoint_alternativo': ckpt_alt}
