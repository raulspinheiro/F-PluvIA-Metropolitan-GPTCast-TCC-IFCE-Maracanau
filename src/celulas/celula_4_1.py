# @title
# Célula 4.1: Validação Fine-Tuned × Zero-Shot — perda de validação de 2024 dos dois modelos, nas mesmas
# janelas, por janela e por posição de quadro, com intervalo de confiança da diferença (bootstrap por dias).
# Responde se o fine-tuning da Célula 4 melhorou o modelo de partida.
#
# Eesuisito de Execução: Rodar depois da Célula 4.
#
import json
import os
import textwrap
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ENV_TAG = globals().get('ENV_TAG', 'Ambiente_Ativo')
ENV_LABEL = globals().get('ENV_LABEL', 'Hardware Ativo')
SEED = int(globals().get('SEED', 42))
N_BOOT = 2000                # reamostragens de dias inteiros
TOL_REGISTRO = 0.02          # tolerância entre a perda medida aqui e a melhor val/loss registrada no treino
TOKENS_QUADRO = 64           # 8 x 8 tokens por quadro

faltam = [n for n in ('gpt8', 'gpt8_zeroshot', 'md_rmf') if n not in globals()]
if faltam:
  raise RuntimeError(f'Rode antes a Célula 4. Faltam: {", ".join(faltam)}')

ds_val = md_rmf.splits['val']
crop = getattr(ds_val, 'crop', globals().get('VAL_CROP', '?'))
n_ctx_frames = int(getattr(ds_val, 'target_seq_len', 8) or 8)
n_win_frames = int(ds_val.sequences_dbz.shape[1])
inicio = {'start': 0, 'end': n_win_frames - n_ctx_frames}.get(crop, 0)
quadros = np.arange(inicio + 1, inicio + n_ctx_frames + 1)           # números dos quadros na janela de 11
CONTEXT = int(globals().get('CONTEXT_STEPS', 7))
previstos = quadros > CONTEXT                                        # quadros 8 a 11 = os previstos na inferência


# ------------------------------------------------------------- aceleração na GPU (CUDA)
# Ao fim do trainer.fit o Lightning devolve o modelo para a CPU, e o checkpoint do Zero-Shot é carregado na CPU.
# Por isso os dois modelos são levados à GPU aqui, antes de medir; eles continuam nela para as células seguintes.
if torch.cuda.is_available():
  for _nome in ('gpt8', 'gpt8_zeroshot'):
    _m = globals()[_nome]
    if _m.device.type != 'cuda':
      print(f'[Célula 4.1] Movendo {_nome} da {_m.device.type.upper()} para a GPU.')
      globals()[_nome] = _m.to('cuda')
  gpt8, gpt8_zeroshot = globals()['gpt8'], globals()['gpt8_zeroshot']
USA_CUDA = torch.cuda.is_available() and gpt8.device.type == 'cuda' and gpt8_zeroshot.device.type == 'cuda'
if torch.cuda.is_available() and not USA_CUDA:
  print('[Aviso Célula 4.1] Não foi possível levar os modelos à GPU: a validação vai rodar na CPU, bem mais devagar.')
if USA_CUDA:
  torch.backends.cuda.matmul.allow_tf32 = True          # operações fora do autocast usam TF32 nos Tensor Cores
  torch.backends.cudnn.allow_tf32 = True
  torch.backends.cudnn.benchmark = True                 # escolhe o algoritmo de convolução mais rápido (formato fixo)
  vram_gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
  # Sem gradientes a memória por janela é bem menor que no treino: lotes maiores aproveitam a GPU.
  BATCH_INF = 64 if vram_gb >= 65 else 32 if vram_gb >= 20 else 8
  _nw = min(8, os.cpu_count() or 2)
  loader_val = DataLoader(ds_val, batch_size=BATCH_INF, shuffle=False, num_workers=_nw, pin_memory=True,
                          persistent_workers=False, prefetch_factor=4)
  torch.cuda.reset_peak_memory_stats(0)
else:
  BATCH_INF = getattr(md_rmf, 'batch_size', '?')
  loader_val = None                                     # usa o carregador da Célula 3


@torch.inference_mode()
def perdas_por_janela(model):
  """Entropia cruzada por token, igual ao shared_step do GPTCast, mas sem fazer a média do lote:
  devolve (janelas, posições de token) para tirar médias por janela e por quadro.
  Na GPU: modo de inferência, autocast bf16, cópia assíncrona da memória fixada e resultados mantidos na GPU
  até o fim (uma única transferência para a CPU)."""
  model.eval()
  dev = model.device
  out = []
  amp = torch.autocast('cuda', dtype=torch.bfloat16) if dev.type == 'cuda' else torch.autocast('cpu', enabled=False)
  for batch in (loader_val if loader_val is not None else md_rmf.val_dataloader()):   # ordem = ds_val.timestamps
    batch = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
    x, c = model.get_xc(batch)
    with amp:
      logits, target = model(x, c)
    ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), target.reshape(-1), reduction='none')
    out.append(ce.view(target.shape))
  return torch.cat(out).cpu().numpy()                    # (N, 512)


def _sync():
  if USA_CUDA:
    torch.cuda.synchronize()
  return time.perf_counter()


t_ini = _sync()
t0 = _sync(); ce_zs = perdas_por_janela(gpt8_zeroshot); t_zs = _sync() - t0
t0 = _sync(); ce_ft = perdas_por_janela(gpt8); t_ft = _sync() - t0
t_total = _sync() - t_ini
pico_vram = torch.cuda.max_memory_allocated(0) / 2 ** 30 if USA_CUDA else 0.0
N, L = ce_ft.shape
if L != n_ctx_frames * TOKENS_QUADRO:
  print(f'[Aviso Célula 4.1] {L} tokens por janela; esperado {n_ctx_frames * TOKENS_QUADRO}.')

win_zs, win_ft = ce_zs.mean(1), ce_ft.mean(1)                        # perda por janela
delta = win_ft - win_zs                                              # Δ_j < 0: ajustado melhor
fr_zs = ce_zs.reshape(N, -1, TOKENS_QUADRO).mean(2)                  # (N, 8) perda por quadro
fr_ft = ce_ft.reshape(N, -1, TOKENS_QUADRO).mean(2)
L_zs, L_ft = float(win_zs.mean()), float(win_ft.mean())
D = L_ft - L_zs

# Bootstrap por dias inteiros (janelas do mesmo dia compartilham quadros)
ts = np.asarray(ds_val.timestamps)[:N]
dias = pd.to_datetime(ts).normalize().values
ud, inv = np.unique(dias, return_inverse=True)
soma_d, cont_d = np.bincount(inv, delta, len(ud)), np.bincount(inv, minlength=len(ud))
rng = np.random.default_rng(SEED)
am = rng.integers(0, len(ud), (N_BOOT, len(ud)))
boot = soma_d[am].sum(1) / cont_d[am].sum(1)
ic = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
frac_melhor = float(np.mean(delta < 0))
D_prev = float((fr_ft[:, previstos] - fr_zs[:, previstos]).mean()) if previstos.any() else np.nan

# Conferência com o registro do treino
rec = {}
if 'training_record_json' in globals() and os.path.isfile(training_record_json):
  rec = json.load(open(training_record_json, encoding='utf-8'))
melhor_reg = rec.get('melhor_val_loss_2024')

# Diagnóstico: o IC 95 % da diferença é o único critério; o ganho em perplexidade vem ao lado, sem limiar.
# Perplexidade = exp(L); a razão FT/ZS é exp(Δ), então o ganho (%) = 100 · (1 − exp(Δ)).
ganho_pp = 100 * (1 - np.exp(D))
ganho_ic = (100 * (1 - np.exp(ic[1])), 100 * (1 - np.exp(ic[0])))
_n = lambda v: f'{v:+.4f}'.replace('.', ',')
_p = lambda v: f'{v:.1f}'.replace('.', ',') + ' %'
_txt_d = f'Δ = {_n(D)} nats por token, IC 95 % de {_n(ic[0])} a {_n(ic[1])}'
if ic[1] < 0:
  diag = (f'O ajustado é melhor que o Zero-Shot ({_txt_d}): perplexidade {_p(ganho_pp)} menor '
          f'(IC 95 % de {_p(ganho_ic[0])} a {_p(ganho_ic[1])}). O checkpoint pode seguir para as Células 5 a 13.1, '
          'onde o ganho precisa aparecer também nas métricas da chuva.')
elif ic[0] > 0:
  diag = (f'O ajustado é PIOR que o Zero-Shot ({_txt_d}): perplexidade {_p(-ganho_pp)} maior '
          f'(IC 95 % de {_p(-ganho_ic[1])} a {_p(-ganho_ic[0])}). O fine-tuning atual não serve: ajuste a Célula 4 '
          'antes de rodar as Células 5 a 13.1.')
else:
  diag = (f'Não há diferença comprovada entre o ajustado e o Zero-Shot ({_txt_d}, que inclui zero): variação de '
          f'perplexidade de {_p(ganho_ic[0])} a {_p(ganho_ic[1])} a favor do ajustado. O fine-tuning atual não trouxe '
          'ganho comprovado: ajuste a Célula 4.')

# ------------------------------------------------------------------- figura
fig = plt.figure(figsize=(16, 6.4))
gs = fig.add_gridspec(1, 2, wspace=0.2)
fig.suptitle('Validação Fine-Tuned × Zero-Shot', fontsize=16, fontweight='bold', y=0.99)

ax1 = fig.add_subplot(gs[0, 0])
for nome, fr, cor, mk in (('Zero-Shot', fr_zs, '#ff7f0e', 'v--'), ('Fine-Tuned', fr_ft, '#d62728', 'o-')):
  m = fr.mean(0)
  ax1.plot(quadros, m, mk, color=cor, lw=2.2, label=f'{nome} (média {fr.mean():.3f})')
if previstos.any():
  ax1.axvspan(quadros[previstos][0] - 0.5, quadros[previstos][-1] + 0.5, color='#1f77b4', alpha=0.08,
              label='Quadros previstos na inferência (8 a 11)')
ax1.set_title(r'Perda por quadro da janela  $(\mathcal{L} = -\frac{1}{N}\,\Sigma_{i=1}^{N}\, \log p_\theta(z_i \mid z_{<i}))$',
              fontweight='bold')
ax1.set_xlabel(f'Quadro da janela de {n_win_frames} (corte de validação: {crop!r})')
ax1.set_ylabel(r'$\mathcal{L}$ (entropia cruzada, nats por token)')
ax1.set_xticks(quadros)
ax1.grid(True, ls=':', alpha=0.6)
ax1.legend(fontsize=8.5)

ax2 = fig.add_subplot(gs[0, 1])
lim = np.percentile(np.abs(delta), 99.5)
ax2.hist(np.clip(delta, -lim, lim), bins=60, color='#7f7f7f', alpha=0.75)
ax2.axvline(0, color='k', lw=1)
ax2.axvline(D, color='#d62728', lw=2.2, label=f'Média Δ = {D:+.4f}')
ax2.axvspan(ic[0], ic[1], color='#d62728', alpha=0.18, label=f'IC 95 % (bootstrap por {len(ud)} dias)')
ax2.set_title(r'Diferença por janela  $(\Delta_j = \mathcal{L}_{\mathrm{FT},j} - \mathcal{L}_{\mathrm{ZS},j})$',
              fontweight='bold')
ax2.set_xlabel(r'$\Delta_j$ (negativo = Fine-Tuned melhor)')
ax2.set_ylabel('Janelas de 2024')
ax2.grid(True, axis='y', ls=':', alpha=0.6)
ax2.legend(fontsize=8.5)

RESULTS_ENV_DIR = globals().get('RESULTS_ENV_DIR') or os.path.join(SERIES_CACHE_DIR, f'resultados_{ENV_TAG}')
os.makedirs(RESULTS_ENV_DIR, exist_ok=True)
png = os.path.join(RESULTS_ENV_DIR, f'validacao_finetuned_vs_zeroshot_{ENV_TAG}.png')
fig.savefig(png, dpi=200, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------- saída de texto
resumo = pd.DataFrame([
    {'Modelo': 'Zero-Shot', 'val/loss (nats/token)': round(L_zs, 4), 'Perplexidade (exp L)': round(float(np.exp(L_zs)), 1),
     'val/loss quadros 8–11': round(float(fr_zs[:, previstos].mean()), 4) if previstos.any() else np.nan},
    {'Modelo': 'Fine-Tuned', 'val/loss (nats/token)': round(L_ft, 4), 'Perplexidade (exp L)': round(float(np.exp(L_ft)), 1),
     'val/loss quadros 8–11': round(float(fr_ft[:, previstos].mean()), 4) if previstos.any() else np.nan},
])
por_quadro = pd.DataFrame({'Quadro': quadros, 'Zero-Shot': fr_zs.mean(0).round(4), 'Fine-Tuned': fr_ft.mean(0).round(4),
                           'Δ (FT − ZS)': (fr_ft.mean(0) - fr_zs.mean(0)).round(4),
                           'Previsto na inferência': np.where(previstos, 'sim', '')})
pd.DataFrame({'t0_fim_janela': ts, 'val_loss_zero_shot': win_zs, 'val_loss_fine_tuned': win_ft, 'delta': delta}).to_csv(
    os.path.join(RESULTS_ENV_DIR, f'validacao_por_janela_finetuned_vs_zeroshot_{ENV_TAG}.csv'), index=False)
resumo.assign(Delta=round(D, 4), IC95_inf=round(ic[0], 4), IC95_sup=round(ic[1], 4),
              Ganho_perplexidade_pct=round(float(ganho_pp), 2), Janelas=N, Dias=len(ud),
              Frac_janelas_FT_melhor=round(frac_melhor, 3)).to_csv(
    os.path.join(RESULTS_ENV_DIR, f'validacao_resumo_finetuned_vs_zeroshot_{ENV_TAG}.csv'), index=False)

print('=== Contexto da validação (F-PluvIA Metropolitan — Célula 4.1) ===')
for ln in (
    f"Ambiente: {ENV_LABEL}  |  dispositivo: {torch.cuda.get_device_name(0) if USA_CUDA else 'CPU'}  |  "
    + (f"CUDA {torch.version.cuda}  |  autocast bf16 + TF32  |  " if USA_CUDA else "sem aceleração  |  ")
    + f"lote de inferência {BATCH_INF}",
    f"Tempo: Zero-Shot {t_zs:.1f} s, Fine-Tuned {t_ft:.1f} s (total {t_total / 60:.1f} min)"
    + (f"  |  pico de VRAM alocada {pico_vram:.1f} GB" if USA_CUDA else ''),
    f"Dados: validação 2024 — {N} janelas em {len(ud)} dias, corte {crop!r} (quadros {quadros[0]} a {quadros[-1]} "
    f"de {n_win_frames}), {L} tokens por janela",
    f"Modelos: Zero-Shot (pesos do Zenodo) × Fine-Tuned (checkpoint "
    f"{os.path.basename(str(globals().get('best_ckpt_path') or rec.get('checkpoint', '?')))})",
    f"Tokenização: quadros empilhados (igual à previsão) nos dois modelos; tokenização do treino registrada: "
    f"{rec.get('tokenizacao_treino', 'não registrada')}",
):
  print('  ' + ln)
print(f'\n[Célula 4.1] Figura -> {png}\n')
print(resumo.set_index('Modelo').to_string())
print(f'\nΔ = val/loss(Fine-Tuned) − val/loss(Zero-Shot) = {D:+.4f} | IC 95 %: {ic[0]:+.4f} a {ic[1]:+.4f} '
      f'(bootstrap de {N_BOOT} reamostragens de {len(ud)} dias)')
print(f'Ganho em perplexidade: {ganho_pp:.1f} % (IC 95 %: {ganho_ic[0]:.1f} a {ganho_ic[1]:.1f} %; positivo = Fine-Tuned melhor)')
print(f'Fine-Tuned melhor em {100 * frac_melhor:.1f} % das {N} janelas' +
      (f' | Δ só nos quadros previstos (8 a 11): {D_prev:+.4f}' if previstos.any() else ''))
print('\n--- Perda média por quadro da janela ---')
print(por_quadro.set_index('Quadro').to_string())
if melhor_reg is not None and abs(L_ft - float(melhor_reg)) > TOL_REGISTRO:
  print(f'\n[Aviso Célula 4.1] A perda do Fine-Tuned medida aqui ({L_ft:.4f}) difere da melhor val/loss registrada no '
        f'treino ({float(melhor_reg):.4f}). Confira se o checkpoint carregado e o corte de validação são os do treino.')
print('\n=== Diagnóstico automático (regras desta célula, não substitui a avaliação das Células 9 a 13.1) ===')
print(textwrap.fill(diag, 110))
print(textwrap.fill('Regra: o IC 95 % da diferença (bootstrap por dias) é o único critério. IC todo abaixo de zero = '
                    'ajustado melhor; todo acima = pior; contendo zero = sem diferença comprovada. O ganho em '
                    'perplexidade, 100 · (1 − e^Δ), é informativo e não tem limiar.', 110))
CELL4_1_VALIDACAO = {'val_loss_zero_shot': L_zs, 'val_loss_fine_tuned': L_ft, 'delta': D, 'ic95': ic,
                     'ganho_perplexidade_pct': float(ganho_pp), 'ganho_perplexidade_ic95_pct': ganho_ic,
                     'janelas': N, 'dias': len(ud)}
