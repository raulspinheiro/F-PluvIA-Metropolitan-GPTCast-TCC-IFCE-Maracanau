# @title
# Célula 7: Densidade Espectral de Potência Radial (RAPSD) a 2 km — bandas definidas pela resolução
# NATIVA do IMERG (~11 km), razão de energia por banda (amplitude) e razão de forma (espectro normalizado).
#
import datetime
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pysteps.utils import spectral

ENV_TAG = globals().get('ENV_TAG', 'Ambiente_Ativo')
ENV_LABEL = globals().get('ENV_LABEL', 'Hardware Ativo')
RESULTS_ENV_DIR = globals().get('RESULTS_ENV_DIR', os.path.join(SERIES_CACHE_DIR, f'resultados_{ENV_TAG}'))
os.makedirs(RESULTS_ENV_DIR, exist_ok=True)
if 'data_str' not in globals():
  data_str = (dt - datetime.timedelta(minutes=30 * FORECAST_STEPS + 180)).strftime('%d/%m/%Y')

# ------------------------------------------------------------ 1. escalas de referência
RES_GRID_KM = float(globals().get('RES_KM', 2.0))
NATIVE_KM = 11.1                       # IMERG 0,1° perto do equador
NYQUIST_GRID_KM = 2 * RES_GRID_KM      # 4 km: limite da malha reamostrada
NYQUIST_NATIVE_KM = 2 * NATIVE_KM      # ~22 km: menor escala que o IMERG de fato resolve
TOKEN_KM = (GRID_SIZE // 8) * RES_GRID_KM   # 32 km: lado de um token do VQGAN
BANDS = {
    f'< {NYQUIST_NATIVE_KM:.0f} km (abaixo da resolução nativa)': (NYQUIST_GRID_KM, NYQUIST_NATIVE_KM),
    f'{NYQUIST_NATIVE_KM:.0f}–64 km (mesoescala)': (NYQUIST_NATIVE_KM, 64.0),
    '> 64 km (regional)': (64.0, np.inf),
}


def compute_frame_rapsd(field_2d, res_km: float = RES_GRID_KM) -> Tuple[np.ndarray, np.ndarray]:
  """RAPSD de um campo (mm/h). Devolve comprimentos de onda (km) e potência, sem a componente DC."""
  f = field_2d.filled(0.0) if isinstance(field_2d, np.ma.MaskedArray) else np.nan_to_num(field_2d, nan=0.0)
  psd, freq = spectral.rapsd(np.clip(f, 0.0, 205.0).astype(np.float64), fft_method=np.fft,
                             return_freq=True, d=res_km, normalize=False)
  ok = freq > 0
  return 1.0 / freq[ok], psd[ok]


def band_ratios(wl, psd_model, psd_obs) -> dict:
  """Energia: soma da potência na banda (modelo / observado). Forma: o mesmo com cada espectro
  dividido pela sua potência total, isolando a distribuição entre escalas da amplitude do campo."""
  out = {}
  tot_m, tot_o = psd_model.sum(), psd_obs.sum()
  for name, (lo, hi) in BANDS.items():
    m = (wl >= lo) & (wl < hi)
    e_o = psd_obs[m].sum()
    out[f'Energia {name}'] = round(float(psd_model[m].sum() / e_o), 3) if e_o > 0 else np.nan
    out[f'Forma {name}'] = (round(float((psd_model[m].sum() / tot_m) / (e_o / tot_o)), 3)
                            if e_o > 0 and tot_m > 0 else np.nan)
  return out


# -------------------------------------------------------------- 2. séries comparadas
spectral_models = {'Observado IMERG-FR': (ground_truth_rain, 'black', '-', 2.6)}
if 'rec_4_rain' in globals():
  spectral_models['VQGAN-8x8 (reconstrução, Estágio 1)'] = (rec_4_rain, '#9467bd', '-.', 1.8)
spectral_models['GPTCast-8x8 ajustado'] = (forecast8_rain, '#d62728', 'o-', 2.0)
if 'forecast8_zeroshot_rain' in globals():
  spectral_models['GPTCast-8x8 zero-shot'] = (forecast8_zeroshot_rain, '#ff7f0e', '--', 1.8)
# Baselines físicos (Lucas-Kanade e LINDA determinístico): usa os da Célula 10 se ela já rodou; senão
# calcula aqui, com exatamente as mesmas configurações da Célula 10, em variáveis locais (_lk7, _linda7).
# Nenhuma variável da Célula 10 é criada nem alterada, então a ordem de execução não importa.
def _baselines_pysteps():
  from pysteps import motion, nowcasts
  from gptcast.utils.converters import dbz_to_rainfall as _d2r
  dbz = np.clip(np.nan_to_num(np.asarray(np.ma.filled(input_data_dbz, 0.0), dtype=np.float64), nan=0.0), 0.0, 60.0)
  rain = np.clip(_d2r(dbz), 0.0, 205.0)
  vel = motion.get_method('LK')(dbz[-4:])                                  # n_motion = 4, em dBZ
  lk = nowcasts.get_method('extrapolation')(rain[-1], vel, FORECAST_STEPS)
  lk = np.clip(np.nan_to_num(lk, nan=0.0), 0.0, 205.0).astype(np.float32)
  try:
    li = nowcasts.get_method('linda')(rain[-3:], vel, FORECAST_STEPS, feature_method='domain', ari_order=1,
                                      add_perturbations=False, kmperpixel=RES_GRID_KM, timestep=30.0,
                                      measure_time=False)
    li = li[0] if isinstance(li, tuple) else li
    li = li[0] if li.ndim == 4 else li
    li = np.clip(np.nan_to_num(li, nan=0.0), 0.0, 205.0).astype(np.float32)
    status = 'LINDA (pysteps)'
  except Exception as e:
    print(f'[Aviso Célula 7] LINDA falhou ({type(e).__name__}); a curva LINDA fica de fora.')
    li, status = None, None
  return lk, li, status


if 'lk_extrap_rain' in globals() and 'linda_rain' in globals():
  _lk7, _linda7, _nome_linda = lk_extrap_rain, linda_rain, 'LINDA (pysteps)'
  _origem_pysteps = 'Célula 10'
else:
  _lk7, _linda7, _nome_linda = _baselines_pysteps()
  _origem_pysteps = 'calculados nesta célula (mesmas configurações da Célula 10)'
spectral_models['Lucas-Kanade (pysteps)'] = (_lk7, '#1f77b4', ':', 1.6)
if _linda7 is not None:
  spectral_models[_nome_linda] = (_linda7, '#2ca02c', ':', 1.6)
print(f'[Célula 7] Lucas-Kanade e LINDA: {_origem_pysteps}.')

# ------------------------------------------------------ 3. estudo de caso, por horizonte
fig, axes = plt.subplots(1, FORECAST_STEPS, figsize=(19, 4.8), constrained_layout=True, sharey=True)
band_records = []
for s in range(FORECAST_STEPS):
  lead = 30 * (s + 1)
  ax = axes[s]
  wl, psd_obs = compute_frame_rapsd(ground_truth_rain[s])
  for name, (seq, color, style, lw) in spectral_models.items():
    wl_m, psd_m = compute_frame_rapsd(seq[s])
    ax.plot(wl_m, 10 * np.log10(np.maximum(psd_m, 1e-10)), style, color=color, lw=lw,
            markersize=3 if 'o' in style else 0, markevery=6, label=name)
    if name == 'Observado IMERG-FR':
      continue
    sel = wl_m >= NYQUIST_NATIVE_KM   # desvio em dB só onde o observado tem informação
    dev = np.abs(10 * np.log10(np.maximum(psd_m[sel], 1e-10)) - 10 * np.log10(np.maximum(psd_obs[sel], 1e-10)))
    band_records.append({'Ambiente_Execucao': ENV_LABEL, 'Hardware_Tag': ENV_TAG, 'Conjunto': f'Estudo de caso {data_str}',
                         'Modelo': name, 'Lead Time': f'+{lead} min', **band_ratios(wl_m, psd_m, psd_obs),
                         f'Desvio Médio ≥ {NYQUIST_NATIVE_KM:.0f} km [dB] (↓)': round(float(dev.mean()), 2)})
  ax.axvspan(NYQUIST_GRID_KM, NYQUIST_NATIVE_KM, color='0.85', alpha=0.6, lw=0,
             label=f'abaixo da resolução nativa (< {NYQUIST_NATIVE_KM:.0f} km)')
  ax.axvline(TOKEN_KM, color='#2ca02c', ls=':', lw=1.4, label=f'lado do token ({TOKEN_KM:.0f} km)')
  ax.set_xscale('log')
  ax.invert_xaxis()
  ax.set_title(f'+{lead} min — {data_str}', fontweight='bold', fontsize=11)
  ax.set_xlabel('Comprimento de onda λ (km)')
  if s == 0:
    ax.set_ylabel('10·log10(PSD) [dB]')
  ax.grid(True, which='both', ls=':', alpha=0.5)
axes[0].legend(loc='lower left', fontsize=7.5)
fig.suptitle(f'RAPSD na RMF (malha de {RES_GRID_KM:.0f} km, IMERG nativo ~{NATIVE_KM:.0f} km) — estudo de caso da '
             f'partição de validação | {ENV_LABEL}', fontsize=12, fontweight='bold')
rapsd_fig_path = os.path.join(RESULTS_ENV_DIR, f'painel_rapsd_espectral_2km_{ENV_TAG}.png')
fig.savefig(rapsd_fig_path, dpi=300, bbox_inches='tight')
plt.show()

# ------------------------------ 4. Estágio 1 em todos os quadros distintos de 2025 (Célula 6)
if 'test_dbz' in globals() and 'rec_t_rain' in globals():
  obs_t = globals().get('obs_t_rain')
  wet = obs_t.max(axis=(1, 2)) >= 1.0
  psd_o = psd_r = None
  for i in np.flatnonzero(wet):
    wl_t, po = compute_frame_rapsd(obs_t[i])
    _, pr = compute_frame_rapsd(rec_t_rain[i])
    psd_o = po if psd_o is None else psd_o + po
    psd_r = pr if psd_r is None else psd_r + pr
  if psd_o is not None:
    band_records.append({'Ambiente_Execucao': ENV_LABEL, 'Hardware_Tag': ENV_TAG,
                         'Conjunto': f'Teste 2025 — {int(wet.sum())} quadros distintos com pico ≥ 1 mm/h',
                         'Modelo': 'VQGAN-8x8 (reconstrução, Estágio 1)', 'Lead Time': '—',
                         **band_ratios(wl_t, psd_r, psd_o)})

df_spectral_raw = pd.DataFrame(band_records)
rapsd_csv_path = os.path.join(RESULTS_ENV_DIR, f'tabela_rapsd_bandas_espectrais_2km_{ENV_TAG}.csv')
df_spectral_raw.to_csv(rapsd_csv_path, index=False)
print(f'=== RAPSD por banda ({ENV_LABEL}) — razão ideal = 1,0 ===')
print('Energia: potência do modelo / observada na banda (mistura amplitude e forma). '
      'Forma: idem com os espectros normalizados pela potência total (só a distribuição entre escalas).\n'
      f'Abaixo de {NYQUIST_NATIVE_KM:.0f} km o observado é suavizado pela reamostragem de ~{NATIVE_KM:.0f} km para '
      f'{RES_GRID_KM:.0f} km: razões ali descrevem textura criada pelo modelo, não fidelidade.')
display(df_spectral_raw.drop(columns=['Hardware_Tag']).set_index(['Ambiente_Execucao', 'Conjunto', 'Modelo', 'Lead Time']))
print(f'[Célula 7] Painel -> {rapsd_fig_path}\n[Célula 7] Tabela -> {rapsd_csv_path}')
