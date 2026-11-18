# @title
# Célula 12.1: Métricas de Classificação (Acurácia, Precisão, Recall/POD, F1, FAR, CSI) por Nível de
# Alerta — por pixel e por MUNICÍPIO da RMF (chuva média na área do município), no teste de 2025
# (Célula 12) e no estudo de caso, com a acurácia trivial de "nunca prever chuva" como referência.
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

ENV_TAG = globals().get('ENV_TAG', 'Ambiente_Ativo')
ENV_LABEL = globals().get('ENV_LABEL', 'Hardware Ativo')
RESULTS_ENV_DIR = globals().get('RESULTS_ENV_DIR', os.path.join(SERIES_CACHE_DIR, f'resultados_{ENV_TAG}'))
os.makedirs(RESULTS_ENV_DIR, exist_ok=True)
ALERT_LEVELS = {0.5: 'Atenção (≥ 0,5 mm/h)', 1.0: 'Alerta (≥ 1,0 mm/h)', 5.0: 'Severo (≥ 5,0 mm/h)'}
LEADS = [30 * (s + 1) for s in range(FORECAST_STEPS)]

if 'CELL12_FULL_2025_RESULTS' not in globals():
  raise RuntimeError('Rode a Célula 12 (versão com execução real) antes desta.')
res_12 = CELL12_FULL_2025_RESULTS
obs_2025 = res_12['obs_rain']
preds_2025 = {'GPTCast-8x8 (Fine-Tuned RMF)': res_12['ft_rain'], 'GPTCast-8x8 (Zero-Shot)': res_12['zs_rain'],
              'LINDA (pysteps)': res_12['linda_rain'], 'Fluxo Óptico LK (pysteps)': res_12['lk_rain'],
              'Persistência (Baseline)': res_12['pers_rain']}
NOME_MEDIA = 'GPTCast-8x8 (Fine-Tuned RMF, média de 8)'
if 'ft_media_rain' in res_12:          # 6º método da Célula 12 (média de 8 sorteios do Fine-Tuned)
  preds_2025 = {'GPTCast-8x8 (Fine-Tuned RMF)': preds_2025['GPTCast-8x8 (Fine-Tuned RMF)'], NOME_MEDIA: res_12['ft_media_rain'],
                **{k: v for k, v in preds_2025.items() if k != 'GPTCast-8x8 (Fine-Tuned RMF)'}}
N_EVAL = int(res_12['n_sequences'])
print(f"[Célula 12.1] Amostragem do GPTCast na Célula 12: {res_12.get('amostragem_texto', 'padrão do forecast() (T=1, sem top-k)')}")
# Rótulos com a decodificação de cada GPTCast, para que tabela e figura se identifiquem sozinhas (o mesmo modelo com
# outra decodificação dá números bem diferentes). Arquivos com sufixo por execução: não sobrescrevem os anteriores.
AMOSTRAGEM_12 = res_12.get('amostragem', 'padrao')
_esc = AMOSTRAGEM_12 == 'escolhida_2024'
ROTULO = {'GPTCast-8x8 (Fine-Tuned RMF)': f"GPTCast-8x8 (Fine-Tuned RMF, {'gulosa' if _esc else 'sorteio padrão'})",
          'GPTCast-8x8 (Zero-Shot)': f"GPTCast-8x8 (Zero-Shot, {'top-k 10' if _esc else 'sorteio padrão'})"}
DECOD_TXT = ('decodificação escolhida na validação de 2024' if _esc else 'decodificação padrão do forecast() (T=1, sem top-k)')
SUF = '' if not _esc else ('_amostragem2024_media8' if 'ft_media_rain' in res_12 else '_amostragem2024')


def classification_scores(tp, fp, fn, tn) -> dict:
  """Métricas em %; NaN quando o denominador é zero (ex.: precisão sem nenhum alerta emitido)."""
  def pct(a, b):
    return 100.0 * a / b if b else np.nan
  ppv, pod = pct(tp, tp + fp), pct(tp, tp + fn)
  return {'Acurácia [%]': pct(tp + tn, tp + fp + fn + tn), 'Precisão / PPV [%]': ppv, 'Recall / POD [%]': pod,
          'F1 [%]': 2 * ppv * pod / (ppv + pod) if np.isfinite(ppv) and np.isfinite(pod) and ppv + pod else np.nan,
          'FAR [%]': pct(fp, tp + fp), 'CSI / IoU [%]': pct(tp, tp + fp + fn),
          'Acurácia trivial "nunca chove" [%]': pct(fp + tn, tp + fp + fn + tn),
          'Taxa base observada [%]': pct(tp + fn, tp + fp + fn + tn)}


def _counts(ob, pb):
  return (float(np.sum(ob & pb)), float(np.sum(~ob & pb)), float(np.sum(ob & ~pb)), float(np.sum(~ob & ~pb)))


# ------------------------------------------------ 1. máscaras dos municípios na malha de 2 km
mun_masks = {}
if globals().get('gdf_rmf') is not None:
  lon_g, lat_g = np.meshgrid(md_rmf.target_lons, md_rmf.target_lats)
  pts = np.column_stack([lon_g.ravel(), lat_g.ravel()])
  for _, r in gdf_rmf.iterrows():
    geom = r.geometry
    polys = list(geom.geoms) if geom.geom_type == 'MultiPolygon' else [geom]
    m = np.zeros(pts.shape[0], bool)
    for pg in polys:
      m |= MplPath(np.asarray(pg.exterior.coords)).contains_points(pts)
    if m.sum() >= 4:   # municípios com ao menos 4 pixels de 2 km (16 km²)
      mun_masks[r['name']] = m.reshape(lon_g.shape)
  print(f'[Célula 12.1] {len(mun_masks)} municípios da RMF com máscara na malha de 2 km.')
else:
  print('[Aviso Célula 12.1] Limites municipais indisponíveis (rode a Célula 5): só a avaliação por pixel será feita.')


def municipal_means(rain_4d):
  """(N, passos, H, W) -> (N, passos, M): chuva média na área de cada município."""
  return np.stack([rain_4d[..., m].mean(axis=-1) for m in mun_masks.values()], axis=-1)


# ---------------------------------------------------------------- 2. tabelas
rows = []
obs_mun = municipal_means(obs_2025) if mun_masks else None
for unidade in (['pixel', 'município'] if mun_masks else ['pixel']):
  o_src = obs_2025 if unidade == 'pixel' else obs_mun
  for name, P in preds_2025.items():
    p_src = P if unidade == 'pixel' else municipal_means(P)
    for thr, nivel in ALERT_LEVELS.items():
      for s, lead in enumerate(LEADS + ['todos']):
        sel = slice(None) if lead == 'todos' else s
        o_b, p_b = o_src[:, sel] >= thr, p_src[:, sel] >= thr
        rows.append({'Conjunto': f'Teste 2025 ({N_EVAL} janelas)', 'Unidade': unidade, 'Nível de alerta': nivel,
                     'Modelo': ROTULO.get(name, name), 'Lead Time': 'média +30 a +120' if lead == 'todos' else f'+{lead} min',
                     **{k: round(v, 2) for k, v in classification_scores(*_counts(o_b, p_b)).items()}})

if 'forecast8_rain' in globals():   # estudo de caso (validação), por pixel
  case = {'GPTCast-8x8 (Fine-Tuned RMF)': forecast8_rain, 'GPTCast-8x8 (Zero-Shot)': globals().get('forecast8_zeroshot_rain'),
          'LINDA (pysteps)': globals().get('linda_rain'), 'Fluxo Óptico LK (pysteps)': globals().get('lk_extrap_rain'),
          'Persistência (Baseline)': globals().get('persistence_rain')}
  for name, P in case.items():
    if P is None:
      continue
    for thr, nivel in ALERT_LEVELS.items():
      rows.append({'Conjunto': f'Estudo de caso {globals().get("data_str", "")} (validação)', 'Unidade': 'pixel',
                   'Nível de alerta': nivel, 'Modelo': name, 'Lead Time': 'média +30 a +120',
                   **{k: round(v, 2) for k, v in classification_scores(*_counts(np.asarray(ground_truth_rain) >= thr,
                                                                                  np.asarray(P) >= thr)).items()}})
df_cls = pd.DataFrame(rows)
cls_csv = os.path.join(RESULTS_ENV_DIR, f'fpluvia_metricas_classificacao_teste2025_{ENV_TAG}_N{N_EVAL}{SUF}.csv')
df_cls.to_csv(cls_csv, index=False)

resumo = df_cls[(df_cls['Lead Time'] == 'média +30 a +120') & df_cls['Conjunto'].str.startswith('Teste')]
cols = ['Unidade', 'Nível de alerta', 'Modelo', 'F1 [%]', 'CSI / IoU [%]', 'Precisão / PPV [%]', 'Recall / POD [%]',
        'FAR [%]', 'Acurácia [%]', 'Acurácia trivial "nunca chove" [%]']
print(f'=== Classificação por nível de alerta — teste 2025 ({N_EVAL} janelas), média de +30 a +120 min | {DECOD_TXT} | {ENV_LABEL} ===')
print('A acurácia por pixel fica perto de 100% até para quem nunca prevê chuva (coluna "trivial"): '
      'use F1, CSI, POD e FAR para comparar.')
with pd.option_context('display.max_rows', None):     # sem o "..." do Colab: mostra todas as linhas
  display(resumo[cols].set_index(['Unidade', 'Nível de alerta', 'Modelo']))

# ---------------------------------------------------------------- 3. painel
styles = [('GPTCast-8x8 (Fine-Tuned RMF)', '#d62728', 'o', '-'), (NOME_MEDIA, '#9467bd', 'P', '-.'),
          ('GPTCast-8x8 (Zero-Shot)', '#ff7f0e', 'v', '--'),
          ('LINDA (pysteps)', '#1f77b4', '^', '-'), ('Fluxo Óptico LK (pysteps)', '#2ca02c', 'd', '--'),
          ('Persistência (Baseline)', '#7f7f7f', 's', ':')]
unidade_plot = 'município' if mun_masks else 'pixel'
panels = [('Precisão / PPV [%]', 'Precisão (↑)'), ('Recall / POD [%]', 'Recall / POD (↑)'),
          ('F1 [%]', 'F1 (↑)'), ('FAR [%]', 'FAR (↓)')]
fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
base = df_cls[(df_cls['Conjunto'].str.startswith('Teste')) & (df_cls['Unidade'] == unidade_plot) &
              (df_cls['Nível de alerta'] == ALERT_LEVELS[0.5]) & (df_cls['Lead Time'] != 'média +30 a +120')]
for ax, (col, title) in zip(axes, panels):
  for name, color, mk, ls in styles:
    sub = base[base['Modelo'] == ROTULO.get(name, name)]
    if sub.empty:                       # método ausente (ex.: Célula 12 sem a média de 8)
      continue
    ax.plot(LEADS, sub[col].values, color=color, marker=mk, ls=ls, lw=2, label=ROTULO.get(name, name))
  ax.set_title(title, fontweight='bold')
  ax.set_xlabel('Horizonte (min)')
  ax.set_ylabel('%')
  ax.set_xticks(LEADS)
  ax.grid(True, ls=':', alpha=0.55)
axes[0].legend(fontsize=8)
fig.suptitle(f'Classificação de alerta ≥ 0,5 mm/h por {unidade_plot} — teste 2025 ({N_EVAL} janelas, fev–mai) | '
             f'{DECOD_TXT} | {ENV_LABEL}',
             fontsize=12, fontweight='bold')
png = os.path.join(RESULTS_ENV_DIR, f'metricas_classificacao_particao_completa_1729_teste_2025_{ENV_TAG}{SUF}.png')
fig.savefig(png, dpi=300, bbox_inches='tight')
plt.show()
print(f'[Célula 12.1] Tabela -> {cls_csv}\n[Célula 12.1] Painel -> {png}')
