# @title
# Célula 6: Estágio 1 — Fidelidade de Reconstrução do VQGAN Congelado a 2 km (Tabela 3):
# estudo de caso + TODOS os quadros distintos da partição de teste de 2025.
#
import datetime
import os
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from gptcast.utils.converters import dbz_to_rainfall
from skimage.metrics import structural_similarity as ssim

ENV_TAG = globals().get('ENV_TAG', 'Ambiente_Ativo')
ENV_LABEL = globals().get('ENV_LABEL', 'Hardware Ativo')
RESULTS_ENV_DIR = globals().get('RESULTS_ENV_DIR', os.path.join(SERIES_CACHE_DIR, f'resultados_{ENV_TAG}'))
os.makedirs(RESULTS_ENV_DIR, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
STEP_MIN = 30
TOKEN_KM = GRID_SIZE // 8 * RES_KM    # 128 px / 8 tokens = 16 px por token = 32 km
THRESHOLDS = (0.5, 1.0, 5.0, 10.0)
BATCH_FRAMES = 128

# Horário do estudo de caso (dt = último quadro da janela, +120 min)
t0_history_end = dt - datetime.timedelta(minutes=STEP_MIN * FORECAST_STEPS)
t0_brt = t0_history_end - datetime.timedelta(hours=3)
data_str = t0_brt.strftime('%d/%m/%Y')


def _sync() -> float:
  if torch.cuda.is_available():
    torch.cuda.synchronize()
  return time.perf_counter()


def _to_rain(dbz: np.ndarray) -> np.ndarray:
  return np.clip(dbz_to_rainfall(np.clip(dbz, 0.0, 60.0).astype(np.float64)), 0.0, 205.0).astype(np.float32)


# --------------------------------------------------------------- 1. reconstrução
@torch.no_grad()
def reconstruct_sequence_vqgan(model_gptcast, seq_dbz, device: str = device,
                               batch_frames: int = BATCH_FRAMES) -> Tuple[np.ndarray, np.ndarray, float]:
  """(S, H, W) em dBZ [0, 60] -> codificador -> codebook (1.024 códigos, 8 x 8 tokens) -> decodificador.

  Devolve (reconstrução em dBZ, em mm/h, tempo de GPU em s), processando em lotes.
  """
  vae = model_gptcast.first_stage_model.to(device)
  vae.eval()
  vdtype = next(vae.parameters()).dtype
  raw = seq_dbz.filled(0.0) if isinstance(seq_dbz, np.ma.MaskedArray) else np.nan_to_num(seq_dbz, nan=0.0)
  raw = np.clip(raw, 0.0, 60.0).astype(np.float32)
  out = np.empty_like(raw)
  t0 = _sync()
  for i in range(0, raw.shape[0], batch_frames):
    x = torch.from_numpy(raw[i:i + batch_frames, None] / 30.0 - 1.0).to(device=device, dtype=vdtype)
    quant, _, _ = vae.encode(x)
    out[i:i + batch_frames] = vae.decode(quant).squeeze(1).float().cpu().numpy()
  t_gpu = _sync() - t0
  rec_dbz = np.clip((out + 1.0) * 30.0, 0.0, 60.0).astype(np.float32)
  return rec_dbz, _to_rain(rec_dbz), float(t_gpu)


# ------------------------------------------------------------------- 2. métricas
def compute_stage1_metrics(obs_rain, rec_rain, obs_dbz, rec_dbz, gpu_time_s: float = 0.0,
                           label: str = 'VQGAN-8x8 (RMF 2 km)') -> Dict:
  """MAE/RMSE (mm/h e dBZ), SSIM em dBZ (faixa fixa de 60 dBZ), PCC, volume, picos, CSI/BIAS."""
  o_r, p_r = obs_rain.astype(np.float64), rec_rain.astype(np.float64)
  o_d, p_d = obs_dbz.astype(np.float64), rec_dbz.astype(np.float64)
  ssim_d = float(np.mean([ssim(o_d[i], p_d[i], data_range=60.0) for i in range(o_d.shape[0])]))
  pcc = float(np.corrcoef(o_r.ravel(), p_r.ravel())[0, 1]) if o_r.std() > 1e-6 and p_r.std() > 1e-6 else np.nan
  row = {
      'Ambiente_Execucao': ENV_LABEL, 'Hardware_Tag': ENV_TAG, 'Conjunto Avaliado': label,
      'Quadros Avaliados': int(o_r.shape[0]),
      'MAE [mm/h] (↓)': round(float(np.mean(np.abs(p_r - o_r))), 4),
      'RMSE [mm/h] (↓)': round(float(np.sqrt(np.mean((p_r - o_r) ** 2))), 4),
      'MAE [dBZ] (↓)': round(float(np.mean(np.abs(p_d - o_d))), 3),
      'RMSE [dBZ] (↓)': round(float(np.sqrt(np.mean((p_d - o_d) ** 2))), 3),
      'SSIM dBZ (↑)': round(ssim_d, 4),
      'PCC (↑)': round(pcc, 4),
      'Razão de Volume Rec/Obs (=1)': round(float(p_r.sum() / max(o_r.sum(), 1e-9)), 3),
      'Pico Obs [mm/h]': round(float(o_r.max()), 2),
      'Pico Rec [mm/h]': round(float(p_r.max()), 2),
  }
  for thr in THRESHOLDS:
    ob, rb = o_r >= thr, p_r >= thr
    tp, fp, fn = int(np.sum(ob & rb)), int(np.sum(~ob & rb)), int(np.sum(ob & ~rb))
    csi = tp / (tp + fp + fn) if tp + fp + fn else np.nan
    bias = (tp + fp) / (tp + fn) if tp + fn else np.nan
    row[f'CSI/BIAS @ {thr} mm/h'] = f'{csi:.3f} / {bias:.2f}' if np.isfinite(csi) else 'sem eventos'
  row['Tempo Reconstrução GPU [s] (↓)'] = round(float(gpu_time_s), 4)
  row['ms por Quadro (↓)'] = round(1000 * gpu_time_s / max(o_r.shape[0], 1), 3)
  return row


# ------------------------------------------------- 3. quadros distintos do teste de 2025
def unique_test_frames(dataset) -> Tuple[np.ndarray, np.ndarray]:
  """As janelas de teste se sobrepõem (passo de 60 min, 11 quadros de 30 min): cada horário aparece
  em várias janelas. Devolve cada quadro de 30 min uma única vez, com o seu horário."""
  S, T = dataset.sequences_dbz, dataset.timestamps
  n_f = S.shape[1]
  offsets = np.array([(k - (n_f - 1)) * STEP_MIN for k in range(n_f)], dtype='timedelta64[m]')
  times = (T[:, None] + offsets[None, :]).ravel()
  _, first = np.unique(times, return_index=True)
  seq_idx, frame_idx = np.divmod(first, n_f)
  return np.clip(S[seq_idx, frame_idx], 0.0, 60.0).astype(np.float32), times[first]


# ------------------------------------------------------------------ 4. avaliação
_ = reconstruct_sequence_vqgan(gpt8, np.zeros((2, GRID_SIZE, GRID_SIZE), np.float32))  # aquecimento

obs_4_dbz = np.clip(np.asarray(ground_truth_dbz.data if hasattr(ground_truth_dbz, 'data') else ground_truth_dbz), 0.0, 60.0)
obs_4_rain = _to_rain(obs_4_dbz)
rec_4_dbz, rec_4_rain, t_vq_case = reconstruct_sequence_vqgan(gpt8, obs_4_dbz)
rows_stage1 = [compute_stage1_metrics(obs_4_rain, rec_4_rain, obs_4_dbz, rec_4_dbz, t_vq_case,
                                      label=f'Estudo de caso {data_str} (validação, 4 quadros alvo)')]

test_dbz, test_times = unique_test_frames(md_rmf.splits['test'])
rec_t_dbz, rec_t_rain, t_vq_test = reconstruct_sequence_vqgan(gpt8, test_dbz)
obs_t_rain = _to_rain(test_dbz)
periodo = f'{str(test_times[0])[:10]} a {str(test_times[-1])[:10]}'
rows_stage1.append(compute_stage1_metrics(
    obs_t_rain, rec_t_rain, test_dbz, rec_t_dbz, t_vq_test,
    label=f'Teste 2025 — todos os quadros distintos ({periodo})'))

wet = (obs_t_rain.max(axis=(1, 2)) >= 1.0)  # quadros com pico observado >= 1 mm/h
if wet.any():
  rows_stage1.append(compute_stage1_metrics(
      obs_t_rain[wet], rec_t_rain[wet], test_dbz[wet], rec_t_dbz[wet], t_vq_test * wet.mean(),
      label='Teste 2025 — quadros com pico observado >= 1 mm/h'))

df_stage1_raw = pd.DataFrame(rows_stage1)
stage1_csv_path = os.path.join(RESULTS_ENV_DIR, f'tabela3_estagio1_vqgan_2km_{ENV_TAG}_2019_2025.csv')
df_stage1_raw.to_csv(stage1_csv_path, index=False)

print(f'=== Tabela 3 — Reconstrução do VQGAN-8x8 congelado ({ENV_LABEL}) ===')
print(f'Cada token cobre {GRID_SIZE // 8} x {GRID_SIZE // 8} pixels = {TOKEN_KM:.0f} x {TOKEN_KM:.0f} km. '
      f'Teste 2025: {len(test_dbz)} quadros distintos de 30 min (de {md_rmf.splits["test"].sequences_dbz.shape[0]} janelas).')
display(df_stage1_raw.drop(columns=['Hardware_Tag']).set_index(['Ambiente_Execucao', 'Conjunto Avaliado']))
print(f'[Célula 6] Tabela 3 salva -> {stage1_csv_path}')

# ------------------------------------------------------------------- 5. mapas
res_abs_rain = np.abs(rec_4_rain - obs_4_rain)
if 'plot_rmf_spatial' in globals():
  plot_rmf_spatial(obs_4_rain, dt, f'Estágio 1: observado IMERG-FR a {RES_KM} km ({data_str}) — referência do VQGAN',
                   save_filename=f'estagio1_01_observado_vqgan_2km_{ENV_TAG}.png')
  plot_rmf_spatial(rec_4_rain, dt, f'Estágio 1: reconstruído pelo VQGAN-8x8 (tokens de {TOKEN_KM:.0f} x {TOKEN_KM:.0f} km) — {data_str}',
                   save_filename=f'estagio1_02_reconstruido_vqgan_2km_{ENV_TAG}.png')
  plot_rmf_spatial(res_abs_rain, dt, f'Estágio 1: resíduo absoluto |reconstruído − observado| em mm/h — {data_str}',
                   save_filename=f'estagio1_03_residuo_absoluto_vqgan_2km_{ENV_TAG}.png')
