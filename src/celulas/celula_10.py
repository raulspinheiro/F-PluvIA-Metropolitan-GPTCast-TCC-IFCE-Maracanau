# @title
# Célula 10: Benchmark de 5 Métodos no Estudo de Caso — GPTCast-8x8 (ajustado e zero-shot) x pysteps
# (Lucas-Kanade e LINDA determinístico) x Persistência, com tempos medidos e ordenação por horizonte.
#
import datetime
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gptcast.utils.converters import dbz_to_rainfall
from pysteps import motion, nowcasts

ENV_TAG = globals().get('ENV_TAG', 'Ambiente_Ativo')
ENV_LABEL = globals().get('ENV_LABEL', 'Hardware Ativo')
RESULTS_ENV_DIR = globals().get('RESULTS_ENV_DIR', os.path.join(SERIES_CACHE_DIR, f'resultados_{ENV_TAG}'))
os.makedirs(RESULTS_ENV_DIR, exist_ok=True)
RES_GRID_KM = float(globals().get('RES_KM', 2.0))
STEP_MIN = 30
if 'data_str' not in globals():
  data_str = (dt - datetime.timedelta(minutes=STEP_MIN * FORECAST_STEPS + 180)).strftime('%d/%m/%Y')


# ------------------------------------------------------------ 1. adapter do pysteps
class PyStepsNowcastAdapter:
  """Lucas-Kanade (campo de movimento + extrapolação semi-lagrangiana) e LINDA (pysteps).

  LINDA roda na versão determinística (add_perturbations=False): uma única previsão reproduzível,
  comparável às demais. O campo de movimento é estimado em dBZ nos últimos `n_motion` quadros.
  """

  def __init__(self, oflow_method: str = 'LK', n_motion: int = 4):
    self.oflow_fn = motion.get_method(oflow_method)
    self.extrap_fn = nowcasts.get_method('extrapolation')
    self.linda_fn = nowcasts.get_method('linda')
    self.n_motion = n_motion
    self.linda_status = 'não executado'

  def compute_motion_field(self, in_dbz_seq) -> np.ndarray:
    seq = np.asarray(np.ma.filled(in_dbz_seq, 0.0), dtype=np.float64)
    return self.oflow_fn(np.clip(np.nan_to_num(seq[-self.n_motion:], nan=0.0), 0.0, 60.0))

  def forecast_extrapolation_lk(self, last_frame_rain, velocity_field, steps: int = 4) -> np.ndarray:
    r = np.clip(np.nan_to_num(last_frame_rain, nan=0.0), 0.0, 205.0).astype(np.float64)
    out = self.extrap_fn(r, velocity_field, steps)
    return np.clip(np.nan_to_num(out, nan=0.0), 0.0, 205.0).astype(np.float32)

  def forecast_linda(self, rain_seq_context, velocity_field, steps: int = 4, res_km: float = RES_GRID_KM,
                     feature_method: str = 'domain', ari_order: int = 1) -> np.ndarray:
    r = np.clip(np.nan_to_num(rain_seq_context[-(ari_order + 2):], nan=0.0), 0.0, 205.0).astype(np.float64)
    try:
      out = self.linda_fn(r, velocity_field, steps, feature_method=feature_method, ari_order=ari_order,
                          add_perturbations=False, kmperpixel=res_km, timestep=float(STEP_MIN),
                          measure_time=False)
      out = out[0] if isinstance(out, tuple) else out
      out = out[0] if out.ndim == 4 else out
      self.linda_status = f'LINDA-{"D" if feature_method == "domain" else "F"} determinístico'
      return np.clip(np.nan_to_num(out, nan=0.0), 0.0, 205.0).astype(np.float32)
    except Exception as e:
      self.linda_status = f'FALHOU ({type(e).__name__}): substituído por Lucas-Kanade'
      print(f'[Aviso Célula 10] LINDA falhou ({e}); a linha "LINDA" será a extrapolação Lucas-Kanade.')
      return self.forecast_extrapolation_lk(rain_seq_context[-1], velocity_field, steps)


# ------------------------------------------------------ 2. métodos físicos, cronometrados
pysteps_adapter = PyStepsNowcastAdapter('LK')
raw_in_dbz_all = np.asarray(np.ma.filled(input_data_dbz, 0.0), dtype=np.float64)
context_rain_all = np.clip(dbz_to_rainfall(np.clip(raw_in_dbz_all, 0.0, 60.0)), 0.0, 205.0)

t0 = time.perf_counter()
velocity_field = pysteps_adapter.compute_motion_field(raw_in_dbz_all)
t_of_s = time.perf_counter() - t0
t0 = time.perf_counter()
lk_extrap_rain = pysteps_adapter.forecast_extrapolation_lk(context_rain_all[-1], velocity_field, FORECAST_STEPS)
t_lk_total_s = t_of_s + time.perf_counter() - t0
t0 = time.perf_counter()
linda_rain = pysteps_adapter.forecast_linda(context_rain_all, velocity_field, FORECAST_STEPS)
t_linda_total_s = t_of_s + time.perf_counter() - t0
t0 = time.perf_counter()
_ = np.repeat(context_rain_all[-1:], FORECAST_STEPS, axis=0)
t_pers_s = time.perf_counter() - t0
speed = np.hypot(velocity_field[0], velocity_field[1])
print(f'[Célula 10] Movimento LK: mediana {np.median(speed) * RES_GRID_KM * 60 / STEP_MIN:.1f} km/h, '
      f'máximo {speed.max() * RES_GRID_KM * 60 / STEP_MIN:.1f} km/h | LK {t_lk_total_s:.3f} s | '
      f'LINDA {t_linda_total_s:.3f} s ({pysteps_adapter.linda_status})')

# ---------------------------------------------------------------- 3. tabela conjunta
t_ft = timing_gpu_log.get('inferencia_4passos_finetuned_s', np.nan) if 'timing_gpu_log' in globals() else np.nan
t_zs = timing_gpu_log.get('inferencia_4passos_zeroshot_s', np.nan) if 'timing_gpu_log' in globals() else np.nan
linha_linda = 'LINDA (pysteps)' if 'FALHOU' not in pysteps_adapter.linda_status else 'LINDA (pysteps) [falhou: é LK]'
blocos = [(df_gpt8, t_ft, 'GPU'),
          (df_gpt8_zs if 'df_gpt8_zs' in globals() else None, t_zs, 'GPU'),
          (evaluate_nowcast_by_leadtime(ground_truth_rain, linda_rain, linha_linda), t_linda_total_s, 'CPU'),
          (evaluate_nowcast_by_leadtime(ground_truth_rain, lk_extrap_rain, 'Fluxo Óptico LK (pysteps)'), t_lk_total_s, 'CPU'),
          (df_pers, t_pers_s, 'CPU')]
parts = []
for df_m, t_m, dev in blocos:
  if df_m is None:
    continue
  d = df_m.copy()
  d['Tempo_Inferência_4Passos [s]'] = round(float(t_m), 4)
  d['Dispositivo'] = dev
  parts.append(d)
df_full_benchmark = pd.concat(parts, ignore_index=True)
bench_csv_path = os.path.join(RESULTS_ENV_DIR, f'benchmark_completo_pysteps_2km_{ENV_TAG}.csv')
df_full_benchmark.to_csv(bench_csv_path, index=False)

# Ordenação por horizonte (1 = melhor) em MAE, CSI ≥ 0,5 e CSI ≥ 1,0 mm/h
rank = df_full_benchmark[['Modelo', 'Lead Time', 'MAE (↓) [mm/h]', 'CSI_0.5', 'CSI_1.0']].copy()
rank['Pos. MAE'] = rank.groupby('Lead Time')['MAE (↓) [mm/h]'].rank(method='min').astype('Int64')
rank['Pos. CSI 0,5'] = rank.groupby('Lead Time')['CSI_0.5'].rank(ascending=False, method='min').astype('Int64')
rank['Pos. CSI 1,0'] = rank.groupby('Lead Time')['CSI_1.0'].rank(ascending=False, method='min').astype('Int64')
df_rank = rank.pivot(index='Modelo', columns='Lead Time', values=['Pos. MAE', 'Pos. CSI 0,5', 'Pos. CSI 1,0'])
_leads = list(dict.fromkeys(df_full_benchmark['Lead Time']))
df_rank = df_rank.reindex(columns=pd.MultiIndex.from_product([['Pos. MAE', 'Pos. CSI 0,5', 'Pos. CSI 1,0'], _leads]))

cols = ['Ambiente_Execucao', 'Modelo', 'Lead Time', 'MAE (↓) [mm/h]', 'RMSE (↓) [mm/h]', 'SSIM (↑)', 'PCC (↑)',
        'Volume Prev/Obs (=1)', 'CSI/BIAS @ 0.5 mm/h', 'CSI/BIAS @ 1.0 mm/h', 'CSI/BIAS @ 5.0 mm/h',
        'Tempo_Inferência_4Passos [s]', 'Dispositivo']
cols = [c for c in cols if c in df_full_benchmark.columns]
print(f'\n=== Benchmark de 5 métodos — estudo de caso {data_str} (validação; 1 evento) | {ENV_LABEL} ===')
print('Tempos: GPTCast em GPU (média de repetições, Célula 4); pysteps e Persistência em CPU, numa execução.')
display(df_full_benchmark[cols].set_index(['Ambiente_Execucao', 'Modelo', 'Lead Time']))
print('\n=== Posição de cada método por horizonte (1 = melhor) ===')
display(df_rank)

# ------------------------------------------------------------------ 4. mapas e painel
if 'plot_rmf_spatial' in globals():
  plot_rmf_spatial(linda_rain, dt, f'{linha_linda} — {pysteps_adapter.linda_status} — {data_str}',
                   save_filename=f'mapa_04_linda_pysteps_2km_{ENV_TAG}.png')
  plot_rmf_spatial(lk_extrap_rain, dt, f'Fluxo óptico Lucas-Kanade (pysteps), extrapolação semi-lagrangiana — {data_str}',
                   save_filename=f'mapa_05_fluxo_optico_lk_2km_{ENV_TAG}.png')

styles = {'GPTCast-8x8 (Fine-Tuned RMF)': ('#d62728', 'o-'), 'GPTCast-8x8 (Zero-Shot)': ('#ff7f0e', 'v--'),
          linha_linda: ('#1f77b4', '^-'), 'Fluxo Óptico LK (pysteps)': ('#2ca02c', 'd--'),
          'Persistência (Baseline)': ('#7f7f7f', 's:')}
metrics = [('RMSE (↓) [mm/h]', 'RMSE (mm/h, ↓)'), ('MAE (↓) [mm/h]', 'MAE (mm/h, ↓)'),
           ('SSIM (↑)', 'SSIM em dBZ (↑)'), ('CSI_0.5', 'CSI ≥ 0,5 mm/h (↑)')]
fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.5), constrained_layout=True)
for name in df_full_benchmark['Modelo'].unique():
  sub = df_full_benchmark[df_full_benchmark['Modelo'] == name]
  color, ls = styles.get(name, ('k', 'o-'))
  for ax, (c, _) in zip(axes, metrics):
    ax.plot(sub['Lead_min_num'], sub[c], ls, color=color, lw=2, label=name)
for ax, (_, title) in zip(axes, metrics):
  ax.set_title(title, fontweight='bold', fontsize=11)
  ax.set_xlabel('Horizonte (min)')
  ax.set_xticks([30, 60, 90, 120])
  ax.grid(True, ls=':', alpha=0.6)
axes[0].legend(fontsize=8)
fig.suptitle(f'Benchmark de 5 métodos — estudo de caso {data_str} (validação) | malha {RES_GRID_KM:.0f} km | {ENV_LABEL}',
             fontsize=12.5, fontweight='bold')
bench_fig_path = os.path.join(RESULTS_ENV_DIR, f'painel_benchmark_5modelos_2km_{ENV_TAG}.png')
fig.savefig(bench_fig_path, dpi=300, bbox_inches='tight')
plt.show()
print(f'[Célula 10] Tabela -> {bench_csv_path}\n[Célula 10] Painel -> {bench_fig_path}')
