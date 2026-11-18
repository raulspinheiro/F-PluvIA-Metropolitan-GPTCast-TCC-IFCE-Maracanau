# @title
# Célula 11: Fractions Skill Score (FSS — Roberts & Lean, 2008) Multiescala a 2 km, até 98 km,
# com escalas abaixo da resolução nativa do IMERG identificadas e casos sem chuva como indefinidos.
#
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

ENV_TAG = globals().get('ENV_TAG', 'Ambiente_Ativo')
ENV_LABEL = globals().get('ENV_LABEL', 'Hardware Ativo')
RESULTS_ENV_DIR = globals().get('RESULTS_ENV_DIR', os.path.join(SERIES_CACHE_DIR, f'resultados_{ENV_TAG}'))
os.makedirs(RESULTS_ENV_DIR, exist_ok=True)
RES_GRID_KM = float(globals().get('RES_KM', 2.0))
NYQUIST_NATIVE_KM = 22.2                    # 2 x ~11,1 km do IMERG
TOKEN_KM = (GRID_SIZE // 8) * RES_GRID_KM   # 32 km
FSS_WINDOWS_PX = [1, 3, 5, 9, 13, 17, 25, 33, 49]   # 2 a 98 km
FSS_THRESHOLDS = (0.5, 1.0)                 # mm/h
fss_windows_px = FSS_WINDOWS_PX
fss_scales_km = [int(round(w * RES_GRID_KM)) for w in FSS_WINDOWS_PX]
if 'data_str' not in globals():
  data_str = (dt - datetime.timedelta(minutes=30 * FORECAST_STEPS + 180)).strftime('%d/%m/%Y')


def compute_fss_2d(obs_2d, pred_2d, thr_mmh: float = 0.5, window_px: int = 1) -> float:
  """FSS para um limiar e uma janela quadrada de `window_px` pixels (borda preenchida com zero).

  Indefinido (NaN) quando nenhum dos dois campos atinge o limiar.
  """
  o = np.nan_to_num(np.ma.filled(obs_2d, 0.0), nan=0.0) >= thr_mmh
  p = np.nan_to_num(np.ma.filled(pred_2d, 0.0), nan=0.0) >= thr_mmh
  if not o.any() and not p.any():
    return np.nan
  fo = uniform_filter(o.astype(np.float64), size=window_px, mode='constant', cval=0.0)
  fp = uniform_filter(p.astype(np.float64), size=window_px, mode='constant', cval=0.0)
  ref = float(np.mean(fo ** 2 + fp ** 2))
  return float(np.clip(1.0 - np.mean((fo - fp) ** 2) / ref, 0.0, 1.0)) if ref > 0 else np.nan


def _useful(obs_2d, thr):
  return 0.5 + float(np.mean(np.nan_to_num(np.ma.filled(obs_2d, 0.0)) >= thr)) / 2.0


def _min_scale(values, useful):
  for v, s in zip(values, fss_scales_km):
    if np.isfinite(v) and v >= useful:
      return f'{s} km'
  return f'> {fss_scales_km[-1]} km'


def fss_over_set(obs_seqs, preds: dict, thr: float = 0.5) -> pd.DataFrame:
  """FSS agregado em muitas janelas (ex.: teste 2025, Célula 12): para cada escala, soma os erros
  quadráticos das frações de todas as janelas antes da razão (forma recomendada por Roberts & Lean
  para conjuntos), por modelo e horizonte. obs_seqs: (N, 4, H, W); preds: nome -> (N, 4, H, W)."""
  rows = []
  for name, P in preds.items():
    for s in range(obs_seqs.shape[1]):
      row = {'Modelo': name, 'Lead Time': f'+{30 * (s + 1)} min', 'Limiar [mm/h]': thr, 'Janelas': obs_seqs.shape[0]}
      f0 = float(np.mean(obs_seqs[:, s] >= thr))
      vals = []
      for w, km in zip(FSS_WINDOWS_PX, fss_scales_km):
        num = den = 0.0
        for i in range(obs_seqs.shape[0]):
          fo = uniform_filter((obs_seqs[i, s] >= thr).astype(np.float64), size=w, mode='constant')
          fp = uniform_filter((P[i, s] >= thr).astype(np.float64), size=w, mode='constant')
          num += np.sum((fo - fp) ** 2)
          den += np.sum(fo ** 2 + fp ** 2)
        v = 1.0 - num / den if den > 0 else np.nan
        vals.append(v)
        row[f'FSS @ {km} km'] = round(float(v), 4)
      row['FSS útil (0,5 + f0/2)'] = round(0.5 + f0 / 2, 4)
      row['Escala Mínima Útil (↓)'] = _min_scale(vals, 0.5 + f0 / 2)
      rows.append(row)
  return pd.DataFrame(rows)


# ------------------------------------------------------------ 1. modelos avaliados
fss_models = {'GPTCast-8x8 (Fine-Tuned RMF)': forecast8_rain}
for nome, var in (('GPTCast-8x8 (Zero-Shot)', 'forecast8_zeroshot_rain'), ('LINDA (pysteps)', 'linda_rain'),
                  ('Fluxo Óptico LK (pysteps)', 'lk_extrap_rain'), ('Persistência (Baseline)', 'persistence_rain')):
  if var in globals():
    fss_models[nome] = globals()[var]

# --------------------------------------------------------- 2. FSS do estudo de caso
fss_rows = []
for thr in FSS_THRESHOLDS:
  for name, pred in fss_models.items():
    for s in range(FORECAST_STEPS):
      useful = _useful(ground_truth_rain[s], thr)
      vals = [compute_fss_2d(ground_truth_rain[s], pred[s], thr, w) for w in FSS_WINDOWS_PX]
      row = {'Ambiente_Execucao': ENV_LABEL, 'Hardware_Tag': ENV_TAG, 'Modelo': name,
             'Lead Time': f'+{30 * (s + 1)} min', 'Lead_min_num': 30 * (s + 1), 'Limiar [mm/h]': thr,
             'FSS_Useful (0.5 + f0/2)': round(useful, 4)}
      for v, km in zip(vals, fss_scales_km):
        row[f'FSS @ {km} km'] = round(v, 4) if np.isfinite(v) else np.nan
      row['Escala Mínima Útil (↓)'] = _min_scale(vals, useful) if np.isfinite(vals).any() else 'indefinido (sem chuva)'
      fss_rows.append(row)
df_fss = pd.DataFrame(fss_rows)
fss_csv_path = os.path.join(RESULTS_ENV_DIR, f'tabela_fss_multiescala_2km_{ENV_TAG}.csv')
df_fss.to_csv(fss_csv_path, index=False)

print(f'=== FSS multiescala — estudo de caso {data_str} (validação; 1 evento) | {ENV_LABEL} ===')
print(f'Escalas abaixo de ~{NYQUIST_NATIVE_KM:.0f} km estão abaixo da resolução nativa do IMERG; '
      f'o lado de um token do VQGAN é {TOKEN_KM:.0f} km. O FSS do teste de 2025 sai da Célula 12 via fss_over_set().')
for thr in FSS_THRESHOLDS:
  sub = df_fss[df_fss['Limiar [mm/h]'] == thr]
  cols = ['Ambiente_Execucao', 'Modelo', 'Lead Time'] + [f'FSS @ {k} km' for k in fss_scales_km] + \
         ['FSS_Useful (0.5 + f0/2)', 'Escala Mínima Útil (↓)']
  print(f'\n--- Limiar ≥ {thr} mm/h ---')
  display(sub[cols].set_index(['Ambiente_Execucao', 'Modelo', 'Lead Time']))

# ------------------------------------------------------------------- 3. painel
palette = {'GPTCast-8x8 (Fine-Tuned RMF)': ('#d62728', 'o-'), 'GPTCast-8x8 (Zero-Shot)': ('#ff7f0e', 'v--'),
           'LINDA (pysteps)': ('#1f77b4', '^-'), 'Fluxo Óptico LK (pysteps)': ('#2ca02c', 'd--'),
           'Persistência (Baseline)': ('#7f7f7f', 's:')}
thr0 = FSS_THRESHOLDS[0]
fig, axes = plt.subplots(1, FORECAST_STEPS, figsize=(19.5, 4.7), constrained_layout=True, sharey=True)
for s, ax in enumerate(axes):
  sub = df_fss[(df_fss['Lead_min_num'] == 30 * (s + 1)) & (df_fss['Limiar [mm/h]'] == thr0)]
  for name in fss_models:
    r = sub[sub['Modelo'] == name]
    if r.empty:
      continue
    color, ls = palette.get(name, ('k', 'o-'))
    ax.plot(fss_scales_km, [r.iloc[0][f'FSS @ {k} km'] for k in fss_scales_km], ls, color=color, lw=2, label=name)
  ax.axhline(sub['FSS_Useful (0.5 + f0/2)'].iloc[0], color='k', ls='--', lw=1, label='FSS útil (0,5 + f0/2)')
  ax.axvspan(fss_scales_km[0], NYQUIST_NATIVE_KM, color='0.88', lw=0, label=f'< {NYQUIST_NATIVE_KM:.0f} km (abaixo do nativo)')
  ax.axvline(TOKEN_KM, color='#2ca02c', ls=':', lw=1.3, label=f'lado do token ({TOKEN_KM:.0f} km)')
  ax.set_title(f'+{30 * (s + 1)} min — limiar ≥ {thr0} mm/h', fontweight='bold', fontsize=11)
  ax.set_xlabel('Escala da vizinhança (km)')
  ax.set_ylim(0, 1)
  ax.grid(True, ls=':', alpha=0.6)
axes[0].set_ylabel('FSS (↑)')
axes[0].legend(fontsize=7.5, loc='lower right')
fig.suptitle(f'FSS multiescala — estudo de caso {data_str} (validação) | malha {RES_GRID_KM:.0f} km | {ENV_LABEL}',
             fontsize=12.5, fontweight='bold')
fss_fig_path = os.path.join(RESULTS_ENV_DIR, f'painel_fss_multiescala_2km_{ENV_TAG}.png')
fig.savefig(fss_fig_path, dpi=300, bbox_inches='tight')
plt.show()
print(f'[Célula 11] Tabela -> {fss_csv_path}\n[Célula 11] Painel -> {fss_fig_path}')
