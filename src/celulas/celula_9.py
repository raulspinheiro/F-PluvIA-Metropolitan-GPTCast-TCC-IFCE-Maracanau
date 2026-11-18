# @title
# Célula 9: Estágio 2 — Métricas por Horizonte (+30 a +120 min) no Estudo de Caso (Tabelas 4 e 5),
# com SSIM em faixa fixa, escores indefinidos marcados como NaN e variabilidade entre sorteios do GPTCast.
#
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gptcast.utils.converters import dbz_to_rainfall, rainfall_to_dbz
from skimage.metrics import structural_similarity as ssim

ENV_TAG = globals().get('ENV_TAG', 'Ambiente_Ativo')
ENV_LABEL = globals().get('ENV_LABEL', 'Hardware Ativo')
RESULTS_ENV_DIR = globals().get('RESULTS_ENV_DIR', os.path.join(SERIES_CACHE_DIR, f'resultados_{ENV_TAG}'))
os.makedirs(RESULTS_ENV_DIR, exist_ok=True)
THRESHOLDS = (0.5, 1.0, 5.0, 10.0)
N_MEMBERS = 10           # sorteios do GPTCast por modelo (a amostragem de tokens é aleatória)
MEMBER_SEEDS = [42 + k for k in range(N_MEMBERS)]   # o membro 42 é o mesmo da Célula 4
if 'data_str' not in globals():
  data_str = (dt - datetime.timedelta(minutes=30 * FORECAST_STEPS + 180)).strftime('%d/%m/%Y')

if 'persistence_rain' not in globals():
  _last = np.asarray(np.ma.filled(input_data_dbz[-1], 0.0))
  last_in_rain = np.clip(dbz_to_rainfall(np.clip(_last, 0.0, 60.0)), 0.0, 205.0)
  persistence_rain = np.repeat(last_in_rain[None], FORECAST_STEPS, axis=0)


def _to_dbz(rain):
  r = np.clip(np.asarray(rain, dtype=np.float64), 0.0, 205.0)
  out = np.zeros_like(r)
  wet = r > 0.0365
  out[wet] = rainfall_to_dbz(r[wet])
  return np.clip(out, 0.0, 60.0)


# -------------------------------------------------- 1. métricas por horizonte (reutilizada)
def evaluate_nowcast_by_leadtime(obs_seq, pred_seq, model_name: str, step_minutes: int = 30,
                                 thresholds: tuple = THRESHOLDS) -> pd.DataFrame:
  """MAE, RMSE, SSIM (em dBZ, faixa fixa de 60 dBZ), PCC, CSI e BIAS por passo futuro.

  Escores sem eventos (ex.: CSI sem chuva acima do limiar em nenhum dos campos) ficam NaN.
  """
  o_all = np.clip(np.nan_to_num(np.asarray(obs_seq), nan=0.0), 0.0, 205.0).astype(np.float64)
  p_all = np.clip(np.nan_to_num(np.asarray(pred_seq), nan=0.0), 0.0, 205.0).astype(np.float64)
  rows = []
  for i in range(o_all.shape[0]):
    o, p = o_all[i], p_all[i]
    row = {
        'Ambiente_Execucao': ENV_LABEL, 'Hardware_Tag': ENV_TAG, 'Modelo': model_name,
        'Lead Time': f'+{(i + 1) * step_minutes} min', 'Lead_min_num': (i + 1) * step_minutes,
        'MAE (↓) [mm/h]': round(float(np.mean(np.abs(p - o))), 4),
        'RMSE (↓) [mm/h]': round(float(np.sqrt(np.mean((p - o) ** 2))), 4),
        'SSIM (↑)': round(float(ssim(_to_dbz(o), _to_dbz(p), data_range=60.0)), 4),
        'PCC (↑)': round(float(np.corrcoef(o.ravel(), p.ravel())[0, 1]), 4) if o.std() > 1e-6 and p.std() > 1e-6 else np.nan,
        'Volume Prev/Obs (=1)': round(float(p.sum() / o.sum()), 3) if o.sum() > 0 else np.nan,
        'Pico Obs [mm/h]': round(float(o.max()), 2), 'Pico Prev [mm/h]': round(float(p.max()), 2),
    }
    for thr in thresholds:
      ob, pb = o >= thr, p >= thr
      tp, fp, fn = int(np.sum(ob & pb)), int(np.sum(~ob & pb)), int(np.sum(ob & ~pb))
      csi = tp / (tp + fp + fn) if tp + fp + fn else np.nan
      bias = (tp + fp) / (tp + fn) if tp + fn else np.nan
      row[f'CSI_{thr}'] = round(csi, 4) if np.isfinite(csi) else np.nan
      row[f'BIAS_{thr}'] = round(bias, 3) if np.isfinite(bias) else np.nan
      if not np.isfinite(csi):
        row[f'CSI/BIAS @ {thr} mm/h'] = 'sem eventos'
      elif not np.isfinite(bias):
        row[f'CSI/BIAS @ {thr} mm/h'] = f'{csi:.3f} / sem obs'
      else:
        row[f'CSI/BIAS @ {thr} mm/h'] = f'{csi:.3f} / {bias:.2f}'
    rows.append(row)
  return pd.DataFrame(rows)


# --------------------------------------------- 2. variabilidade entre sorteios do GPTCast
def forecast_members(model, seeds):
  """N previsões do mesmo contexto, uma por semente: (N, passos, H, W) em mm/h."""
  out = []
  for sd in seeds:
    torch.manual_seed(sd)
    np.random.seed(sd)
    raw = model.forecast(clean_in_dbz, steps=FORECAST_STEPS, units='dbz', verbosity=0)
    raw = raw.filled(0.0) if isinstance(raw, np.ma.MaskedArray) else np.asarray(raw)
    out.append(dbz_to_rainfall(np.clip(np.nan_to_num(raw, nan=0.0), 0.0, 60.0)))
  return np.stack(out).astype(np.float32)


members = {'GPTCast-8x8 (Fine-Tuned RMF)': forecast_members(gpt8, MEMBER_SEEDS)}
if 'gpt8_zeroshot' in globals():
  members['GPTCast-8x8 (Zero-Shot)'] = forecast_members(gpt8_zeroshot, MEMBER_SEEDS)

# ------------------------------------------------------------ 3. tabelas 4 e 5
df_gpt8 = evaluate_nowcast_by_leadtime(ground_truth_rain, forecast8_rain, 'GPTCast-8x8 (Fine-Tuned RMF)')
df_pers = evaluate_nowcast_by_leadtime(ground_truth_rain, persistence_rain, 'Persistência (Baseline)')
dfs = [df_gpt8]
if 'forecast8_zeroshot_rain' in globals():
  df_gpt8_zs = evaluate_nowcast_by_leadtime(ground_truth_rain, forecast8_zeroshot_rain, 'GPTCast-8x8 (Zero-Shot)')
  dfs.append(df_gpt8_zs)
dfs.append(df_pers)
for name, M in members.items():   # média do conjunto de sorteios como previsão adicional
  dfs.append(evaluate_nowcast_by_leadtime(ground_truth_rain, M.mean(axis=0), f'{name} — média de {N_MEMBERS} sorteios'))
df_stage2_eval = pd.concat(dfs, ignore_index=True)
stage2_csv_path = os.path.join(RESULTS_ENV_DIR, f'tabelas4e5_estagio2_leadtime_2km_{ENV_TAG}_2019_2025.csv')
df_stage2_eval.to_csv(stage2_csv_path, index=False)

# Métricas de cada sorteio (calculadas uma vez e usadas na tabela, no ranking de sementes e no painel)
PAINEL = [('RMSE (↓) [mm/h]', 'RMSE (mm/h, ↓)', False), ('MAE (↓) [mm/h]', 'MAE (mm/h, ↓)', False),
          ('SSIM (↑)', 'SSIM em dBZ (↑)', True), ('CSI_0.5', 'CSI ≥ 0,5 mm/h (↑)', True)]
TAB_COLS = ['MAE (↓) [mm/h]', 'RMSE (↓) [mm/h]', 'SSIM (↑)', 'CSI_0.5', 'CSI_1.0', 'Pico Prev [mm/h]', 'Volume Prev/Obs (=1)']
per_member = {name: [evaluate_nowcast_by_leadtime(ground_truth_rain, M[k], name) for k in range(len(M))]
              for name, M in members.items()}
vals = {name: {c: np.array([pd.to_numeric(d[c], errors='coerce').values for d in dfs], dtype=float)
               for c in TAB_COLS} for name, dfs in per_member.items()}       # (sorteios, horizontes)
leads_txt = list(per_member[next(iter(per_member))][0]['Lead Time'])

# Dispersão das métricas entre os sorteios (média ± desvio-padrão)
spread_rows = []
for name, V in vals.items():
  for j, lt in enumerate(leads_txt):
    r = {'Modelo': name, 'Lead Time': lt, 'Sorteios': len(MEMBER_SEEDS)}
    for c in ('MAE (↓) [mm/h]', 'SSIM (↑)', 'CSI_0.5', 'CSI_1.0', 'Pico Prev [mm/h]', 'Volume Prev/Obs (=1)'):
      r[c] = f'{np.nanmean(V[c][:, j]):.3f} ± {np.nanstd(V[c][:, j], ddof=1):.3f}'
    spread_rows.append(r)
df_stage2_spread = pd.DataFrame(spread_rows)
df_stage2_spread.to_csv(os.path.join(RESULTS_ENV_DIR, f'tabela_estagio2_dispersao_sorteios_{ENV_TAG}.csv'), index=False)

# Melhor e pior semente por modelo, métrica e horizonte
seed_rows = []
for name, V in vals.items():
  for c, titulo, maior_melhor in PAINEL:
    for j, lt in enumerate(leads_txt):
      v = V[c][:, j]
      if np.all(np.isnan(v)):
        continue
      ib, iw = (np.nanargmax(v), np.nanargmin(v)) if maior_melhor else (np.nanargmin(v), np.nanargmax(v))
      seed_rows.append({'Modelo': name, 'Métrica': titulo, 'Lead Time': lt,
                        'Média ± dp': f'{np.nanmean(v):.3f} ± {np.nanstd(v, ddof=1):.3f}',
                        'Melhor (semente)': f'{v[ib]:.3f} ({MEMBER_SEEDS[ib]})',
                        'Pior (semente)': f'{v[iw]:.3f} ({MEMBER_SEEDS[iw]})',
                        'Semente 42': f'{v[MEMBER_SEEDS.index(42)]:.3f}' if 42 in MEMBER_SEEDS else '—'})
df_stage2_seeds = pd.DataFrame(seed_rows)
df_stage2_seeds.to_csv(os.path.join(RESULTS_ENV_DIR, f'tabela_estagio2_melhor_pior_semente_{ENV_TAG}.csv'), index=False)

cols = ['Ambiente_Execucao', 'Modelo', 'Lead Time', 'MAE (↓) [mm/h]', 'RMSE (↓) [mm/h]', 'SSIM (↑)', 'PCC (↑)',
        'Volume Prev/Obs (=1)'] + [f'CSI/BIAS @ {t} mm/h' for t in THRESHOLDS] + ['Pico Obs [mm/h]', 'Pico Prev [mm/h]']
print(f'=== Tabelas 4 e 5 — Estágio 2 por horizonte no estudo de caso {data_str} (validação) | {ENV_LABEL} ===')
print('SSIM em dBZ com faixa fixa de 60 dBZ. "sem eventos" = nenhum pixel acima do limiar em nenhum dos campos.')
display(df_stage2_eval[cols].set_index(['Ambiente_Execucao', 'Modelo', 'Lead Time']))
print(f'\n=== Variabilidade entre {N_MEMBERS} sorteios do GPTCast (mesmo contexto, sementes {MEMBER_SEEDS[0]}–{MEMBER_SEEDS[-1]}) ===')
display(df_stage2_spread.set_index(['Modelo', 'Lead Time']))
print('\n=== Melhor e pior sorteio por métrica e horizonte (semente entre parênteses) ===')
print('Para ver o mapa de um sorteio: torch.manual_seed(semente); np.random.seed(semente); model.forecast(...)')
display(df_stage2_seeds.set_index(['Modelo', 'Métrica', 'Lead Time']))

# ------------------------------------------------------------------ 4. painel
# Para cada GPTCast: faixa clara = mín.–máx.; faixa escura = média ± 1 dp;
# linha grossa = média dos sorteios; a semente do pior sorteio de cada horizonte é escrita ao lado do ponto.
fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), constrained_layout=True)
cores = {'GPTCast-8x8 (Fine-Tuned RMF)': ('#d62728', 'o-'), 'GPTCast-8x8 (Zero-Shot)': ('#ff7f0e', 'v--')}
x = np.arange(1, FORECAST_STEPS + 1) * 30
for name, V in vals.items():
  color, ls = cores.get(name, ('#1f77b4', 'o-'))
  for ax, (c, _, maior_melhor) in zip(axes, PAINEL):
    v = V[c]
    med, dp = np.nanmean(v, 0), np.nanstd(v, 0, ddof=1)
    ax.fill_between(x, np.nanmin(v, 0), np.nanmax(v, 0), color=color, alpha=0.08, lw=0, zorder=0)
    ax.fill_between(x, med - dp, med + dp, color=color, alpha=0.22, lw=0, zorder=1)
    ax.plot(x, med, ls, color=color, lw=2.2, zorder=3, label=f'{name} — média de {len(v)} sorteios')
    for j in range(len(x)):          # semente do pior sorteio, só onde ele se afasta da média
      col = v[:, j]
      if np.all(np.isnan(col)):
        continue
      iw = np.nanargmin(col) if maior_melhor else np.nanargmax(col)
      if abs(col[iw] - med[j]) > 1.5 * (dp[j] if dp[j] > 0 else np.inf):
        ax.annotate(f's{MEMBER_SEEDS[iw]}', (x[j], col[iw]), xytext=(4, 0), textcoords='offset points',
                    fontsize=7.5, color=color, fontweight='bold', va='center')
sub = df_stage2_eval[df_stage2_eval['Modelo'] == 'Persistência (Baseline)']
for ax, (c, titulo, _) in zip(axes, PAINEL):
  if not sub.empty:
    ax.plot(sub['Lead_min_num'], pd.to_numeric(sub[c], errors='coerce'), 's:', color='#7f7f7f', lw=2,
            zorder=4, label='Persistência (Baseline)')
  ax.set_title(titulo, fontweight='bold', fontsize=11)
  ax.set_xlabel('Horizonte (min)')
  ax.set_xticks(x)
  ax.grid(True, ls=':', alpha=0.6)
axes[0].legend(fontsize=7.5)
fig.suptitle(f'Estágio 2 por horizonte — estudo de caso {data_str} (validação) | linha grossa = média de '
             f'{len(MEMBER_SEEDS)} sorteios, faixa escura = ± 1 dp, faixa clara = mín.–máx., sNN = semente do pior '
             f'sorteio | {ENV_LABEL}', fontsize=11, fontweight='bold')
stage2_fig_path = os.path.join(RESULTS_ENV_DIR, f'painel_estagio2_leadtime_2km_{ENV_TAG}.png')
fig.savefig(stage2_fig_path, dpi=300, bbox_inches='tight')
plt.show()
print(f'[Célula 9] Tabelas -> {stage2_csv_path}\n[Célula 9] Painel -> {stage2_fig_path}')
