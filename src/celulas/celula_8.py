# @title
# Célula 8: Verificação Orientada a Objetos — SAL (Wernli et al., 2008) a 2 km, com casos sem chuva
# tratados como indefinidos, razão de volume explícita e função reutilizável para conjuntos (Célula 12).
#
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass, label
from gptcast.utils.converters import dbz_to_rainfall

ENV_TAG = globals().get('ENV_TAG', 'Ambiente_Ativo')
ENV_LABEL = globals().get('ENV_LABEL', 'Hardware Ativo')
RESULTS_ENV_DIR = globals().get('RESULTS_ENV_DIR', os.path.join(SERIES_CACHE_DIR, f'resultados_{ENV_TAG}'))
os.makedirs(RESULTS_ENV_DIR, exist_ok=True)
RES_GRID_KM = float(globals().get('RES_KM', 2.0))
SAL_MIN_RAIN = 0.1          # mm/h: pixel com chuva
SAL_OBJ_FRACTION = 1 / 15   # limiar de objeto = R95 / 15 (Wernli et al., 2009)
if 'data_str' not in globals():
  data_str = (dt - datetime.timedelta(minutes=30 * FORECAST_STEPS + 180)).strftime('%d/%m/%Y')


def _clean(f):
  f = f.filled(0.0) if isinstance(f, np.ma.MaskedArray) else np.nan_to_num(f, nan=0.0)
  return np.clip(f, 0.0, 205.0).astype(np.float64)


def _objects(field):
  """Objetos contíguos acima de R95/15: centro de massa global, dispersão r e volume escalado V."""
  wet = field[field >= SAL_MIN_RAIN]
  if wet.size == 0:
    return None
  thr = max(SAL_MIN_RAIN, np.percentile(wet, 95) * SAL_OBJ_FRACTION)
  lab, n = label(field >= thr)
  cm_glob = np.array(center_of_mass(field))
  num_r = num_v = mass = 0.0
  for k in range(1, n + 1):
    m = lab == k
    r_n = float(field[m].sum())
    v_n = r_n / float(field[m].max())
    d_n = float(np.linalg.norm(np.array(center_of_mass(field * m)) - cm_glob))
    num_r += r_n * d_n
    num_v += r_n * v_n
    mass += r_n
  return {'cm': cm_glob, 'r': num_r / mass, 'V': num_v / mass, 'n_obj': n}


def compute_sal_step(obs_2d, mod_2d, res_km: float = RES_GRID_KM, min_rain_thr: float = SAL_MIN_RAIN) -> dict:
  """S, A, L (Wernli et al., 2008) e deslocamento do centro de massa em km.

  SAL não é definido quando um dos campos não tem chuva: devolve NaN e o motivo.
  """
  R_o, R_m = _clean(obs_2d), _clean(mod_2d)
  out = {'S': np.nan, 'A': np.nan, 'L': np.nan, 'SAL_abs': np.nan, 'Shift_km': np.nan,
         'Razao_Volume': np.nan, 'N_obj_obs': 0, 'N_obj_mod': 0, 'Status': ''}
  o, m = _objects(R_o), _objects(R_m)
  if o is None or m is None:
    out['Status'] = 'indefinido: ' + ('sem chuva observada' if o is None else 'sem chuva prevista')
    return out
  h, w = R_o.shape
  d_max = float(np.hypot(h - 1, w - 1))
  mo, mm = R_o.mean(), R_m.mean()
  A = (mm - mo) / (0.5 * (mm + mo))
  S = (m['V'] - o['V']) / (0.5 * (m['V'] + o['V']))
  shift_px = float(np.linalg.norm(m['cm'] - o['cm']))
  L = min(2.0, shift_px / d_max + 2.0 * abs(m['r'] - o['r']) / d_max)
  out.update({'S': round(float(S), 4), 'A': round(float(A), 4), 'L': round(float(L), 4),
              'SAL_abs': round(float(abs(S) + abs(A) + L), 4), 'Shift_km': round(shift_px * res_km, 2),
              'Razao_Volume': round(float(mm / mo), 3), 'N_obj_obs': o['n_obj'], 'N_obj_mod': m['n_obj'],
              'Status': 'ok'})
  return out


def sal_over_set(obs_seqs, preds: dict) -> pd.DataFrame:
  """SAL em muitas janelas (ex.: teste 2025 na Célula 12): mediana e intervalo interquartil por modelo
  e horizonte, mais a contagem de casos indefinidos. obs_seqs: (N, 4, H, W); preds: nome -> (N, 4, H, W)."""
  rows = []
  for name, P in preds.items():
    for s in range(obs_seqs.shape[1]):
      r = pd.DataFrame([compute_sal_step(obs_seqs[i, s], P[i, s]) for i in range(obs_seqs.shape[0])])
      ok = r[r['Status'] == 'ok']
      row = {'Modelo': name, 'Lead Time': f'+{30 * (s + 1)} min', 'Janelas': len(r), 'Indefinidas': int(len(r) - len(ok))}
      for c in ('S', 'A', 'L', 'Shift_km', 'SAL_abs', 'Razao_Volume'):
        q = ok[c].quantile([0.25, 0.5, 0.75]) if len(ok) else pd.Series([np.nan] * 3, index=[0.25, 0.5, 0.75])
        row[f'{c} mediana'] = round(float(q[0.5]), 3)
        row[f'{c} IQR'] = f'{q[0.25]:.2f} a {q[0.75]:.2f}'
      rows.append(row)
  return pd.DataFrame(rows)


# --------------------------------------------------------- 1. persistência (baseline)
last_in_dbz = input_data_dbz[-1].filled(0.0) if isinstance(input_data_dbz[-1], np.ma.MaskedArray) else np.nan_to_num(input_data_dbz[-1], nan=0.0)
last_in_rain = np.clip(dbz_to_rainfall(np.clip(last_in_dbz, 0.0, 60.0)), 0.0, 205.0)
persistence_rain = np.repeat(last_in_rain[None], FORECAST_STEPS, axis=0)

# ----------------------------------------------------------- 2. SAL do estudo de caso
sal_models_dict = {'GPTCast-8x8 ajustado': forecast8_rain}
if 'forecast8_zeroshot_rain' in globals():
  sal_models_dict['GPTCast-8x8 zero-shot'] = forecast8_zeroshot_rain
sal_models_dict['Persistência'] = persistence_rain
for nome, var in (('Lucas-Kanade (pysteps)', 'lk_extrap_rain'), ('LINDA (pysteps)', 'linda_rain')):
  if var in globals():   # entram se a Célula 10 já rodou nesta sessão
    sal_models_dict[nome] = globals()[var]

sal_records = []
for name, pred in sal_models_dict.items():
  for s in range(FORECAST_STEPS):
    r = compute_sal_step(ground_truth_rain[s], pred[s])
    sal_records.append({
        'Ambiente_Execucao': ENV_LABEL, 'Hardware_Tag': ENV_TAG, 'Modelo': name,
        'Lead Time': f'+{30 * (s + 1)} min', 'Lead_num': 30 * (s + 1),
        'Estrutura (S)': r['S'], 'Amplitude (A)': r['A'], 'Localização (L) [↓]': r['L'],
        'Deslocamento Baricentro [km] (↓)': r['Shift_km'], '|S| + |A| + L (↓)': r['SAL_abs'],
        'Volume Previsto / Observado (=1)': r['Razao_Volume'],
        'Objetos Obs / Prev': f"{r['N_obj_obs']} / {r['N_obj_mod']}", 'Situação': r['Status'],
    })
df_sal = pd.DataFrame(sal_records)
sal_csv_path = os.path.join(RESULTS_ENV_DIR, f'tabela_sal_objetos_2km_{ENV_TAG}.csv')
df_sal.to_csv(sal_csv_path, index=False)

melhor = df_sal.groupby('Lead Time', sort=False)['|S| + |A| + L (↓)'].idxmin()
print(f'=== SAL no estudo de caso {data_str} (partição de validação; 1 quadro por horizonte) | {ENV_LABEL} ===')
print('Menor |S|+|A|+L por horizonte: ' + '; '.join(f"{lt}: {df_sal.loc[i, 'Modelo']}" for lt, i in melhor.items()))
print('Um único evento não permite generalizar; o SAL do teste de 2025 sai da Célula 12 via sal_over_set().')
display(df_sal.drop(columns=['Lead_num', 'Hardware_Tag']).set_index(['Ambiente_Execucao', 'Modelo', 'Lead Time']))

# ------------------------------------------------------------------- 3. diagrama SAL
n = len(sal_models_dict)
fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 5.4), constrained_layout=True, squeeze=False)
for ax, name in zip(axes[0], sal_models_dict):
  sub = df_sal[(df_sal['Modelo'] == name) & (df_sal['Situação'] == 'ok')]
  ax.axhline(0, color='k', lw=0.9, alpha=0.6)
  ax.axvline(0, color='k', lw=0.9, alpha=0.6)
  ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, 1, facecolor='#e6f5d0', edgecolor='#4d9221', ls='--',
                             alpha=0.35, zorder=1, label='|S|, |A| ≤ 0,5'))
  ax.plot(sub['Estrutura (S)'], sub['Amplitude (A)'], color='gray', ls=':', lw=1.2, zorder=2)
  sc = ax.scatter(sub['Estrutura (S)'], sub['Amplitude (A)'], c=sub['Localização (L) [↓]'], cmap='plasma_r',
                  vmin=0, vmax=0.5, s=120, edgecolors='k', linewidths=1, zorder=3)
  for _, r in sub.iterrows():
    ax.annotate(f"+{int(r['Lead_num'])} min\n{r['Deslocamento Baricentro [km] (↓)']:.0f} km · vol ×{r['Volume Previsto / Observado (=1)']:.2f}",
                (r['Estrutura (S)'], r['Amplitude (A)']), textcoords='offset points', xytext=(6, 5), fontsize=7.8)
  ax.set_xlim(-2.05, 2.05)
  ax.set_ylim(-2.05, 2.05)
  ax.set_title(name, fontweight='bold', fontsize=11)
  ax.set_xlabel('S (← objetos pequenos/picudos | grandes/achatados →)')
  ax.set_ylabel('A (← subestima | superestima →)')
  ax.grid(True, ls=':', alpha=0.5)
  ax.legend(loc='lower right', fontsize=8)
cb = fig.colorbar(sc, ax=axes[0].tolist(), fraction=0.02, pad=0.02)
cb.set_label('L (0 = posição perfeita)')
fig.suptitle(f'Diagrama SAL — estudo de caso {data_str} (validação) | malha {RES_GRID_KM:.0f} km | {ENV_LABEL}',
             fontsize=12, fontweight='bold')
sal_fig_path = os.path.join(RESULTS_ENV_DIR, f'diagrama_sal_2km_{ENV_TAG}.png')
fig.savefig(sal_fig_path, dpi=300, bbox_inches='tight')
plt.show()
print(f'[Célula 8] Tabela -> {sal_csv_path}\n[Célula 8] Diagrama -> {sal_fig_path}')
