# @title
# Célula 13.2: Métricas de Classificação por Nível de Alerta no TESTE CEGO DE 2026 (Célula 13.1), por pixel e por
# MUNICÍPIO da RMF, lado a lado com o teste de 2025 (Célula 12) — o equivalente da Célula 12.1 para 2026.
# Também sorteia os dias de novo, mil vezes, para ver se cada vantagem no F1 é de verdade ou só sorte (IC de 95 %).
# Não roda inferência: usa CELL13_1_RESULTS e CELL12_FULL_2025_RESULTS já em memória.
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

ENV_TAG = globals().get('ENV_TAG', 'Ambiente_Ativo')
ENV_LABEL = globals().get('ENV_LABEL', 'Hardware Ativo')
RESULTS_ENV_DIR = globals().get('RESULTS_ENV_DIR', os.path.join(SERIES_CACHE_DIR, f'resultados_{ENV_TAG}'))
OUT_DIR = os.path.join(RESULTS_ENV_DIR, 'teste_cego_2026')
os.makedirs(OUT_DIR, exist_ok=True)
ALERT_LEVELS = {0.5: 'Atenção (≥ 0,5 mm/h)', 1.0: 'Alerta (≥ 1,0 mm/h)', 5.0: 'Severo (≥ 5,0 mm/h)'}
LEADS = [30 * (s + 1) for s in range(FORECAST_STEPS)]
N_BOOT = 1000                                   # reamostragens de dias inteiros
SEED = int(globals().get('SEED', 42))

if 'CELL13_1_RESULTS' not in globals():
  raise RuntimeError('Rode a Célula 13.1 antes desta (nesta mesma sessão).')
R13 = CELL13_1_RESULTS
R12 = globals().get('CELL12_FULL_2025_RESULTS')
if R13['parcial']:
  print(f"[Aviso Célula 13.2] A Célula 13.1 terminou PARCIAL ({R13['janelas_prontas']} de {R13['janelas_total']} janelas): "
        'os números abaixo são de uma amostra aleatória da série.')

# --------------------------------------------- 1. métodos, rótulos e sufixo (os mesmos da Célula 13.1)
_esc = R13.get('amostragem', 'padrao') == 'escolhida_2024'
NOMES = {'ft': f"GPTCast-8x8 (Fine-Tuned RMF, {'gulosa' if _esc else 'sorteio padrão'})",
         'ft_media': 'GPTCast-8x8 (Fine-Tuned RMF, média de 8)',
         'zs': f"GPTCast-8x8 (Zero-Shot, {'top-k 10' if _esc else 'sorteio padrão'})",
         'linda': 'LINDA (pysteps)', 'lk': 'Fluxo Óptico LK (pysteps)', 'pers': 'Persistência (Baseline)'}
NOMES = {k: v for k, v in NOMES.items() if f'{k}_rain' in R13}
SUF = '' if not _esc else ('_amostragem2024_media8' if 'ft_media' in NOMES else '_amostragem2024')
DECOD_TXT = 'decodificação escolhida na validação de 2024' if _esc else 'decodificação padrão do forecast()'
N26 = int(R13['janelas_prontas'])
PROD26 = ', '.join(sorted(set(np.asarray(R13['produto']).tolist())))

conjuntos = {f'2026 ({N26} janelas, {PROD26})': (R13['obs_rain'], {k: R13[f'{k}_rain'] for k in NOMES})}
datas = {list(conjuntos)[0]: np.asarray(R13['timestamps_t0'])}
if R12 is not None:
  if R12.get('amostragem', 'padrao') != R13.get('amostragem', 'padrao'):
    print('[Aviso Célula 13.2] Células 12 e 13.1 com decodificações diferentes: a comparação 2025 x 2026 as mistura.')
  m25 = np.isin(pd.to_datetime(np.asarray(R12['timestamps_t0'])).month, [2, 3, 4, 5])
  conjuntos[f'2025 ({int(m25.sum())} janelas, IMERG-FR)'] = (
      R12['obs_rain'][m25], {k: R12[f'{k}_rain'][m25] for k in NOMES if f'{k}_rain' in R12})
  datas[list(conjuntos)[1]] = np.asarray(R12['timestamps_t0'])[m25]
else:
  print('[Célula 13.2] CELL12_FULL_2025_RESULTS ausente: só 2026 (rode a Célula 12 para comparar com 2025).')
ROTULO_2026, ROTULO_2025 = list(conjuntos)[0], (list(conjuntos)[1] if len(conjuntos) > 1 else None)


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


# ------------------------------------------------ 2. máscaras dos municípios na malha de 2 km
mun_masks = globals().get('mun_masks') or {}
if not mun_masks and globals().get('gdf_rmf') is not None:      # a Célula 12.1 pode não ter rodado nesta sessão
  _grade = globals().get('md_rmf') or adaptee
  lon_g, lat_g = np.meshgrid(_grade.target_lons, _grade.target_lats)
  pts = np.column_stack([lon_g.ravel(), lat_g.ravel()])
  for _, r in gdf_rmf.iterrows():
    geom = r.geometry
    polys = list(geom.geoms) if geom.geom_type == 'MultiPolygon' else [geom]
    m = np.zeros(pts.shape[0], bool)
    for pg in polys:
      m |= MplPath(np.asarray(pg.exterior.coords)).contains_points(pts)
    if m.sum() >= 4:   # municípios com ao menos 4 pixels de 2 km (16 km²)
      mun_masks[r['name']] = m.reshape(lon_g.shape)
if mun_masks:
  print(f'[Célula 13.2] {len(mun_masks)} municípios da RMF com máscara na malha de 2 km.')
else:
  print('[Aviso Célula 13.2] Limites municipais indisponíveis (rode a Célula 5): só a avaliação por pixel será feita.')


def municipal_means(rain_4d):
  """(N, passos, H, W) -> (N, passos, M): chuva média na área de cada município."""
  return np.stack([rain_4d[..., m].mean(axis=-1) for m in mun_masks.values()], axis=-1)


# ---------------------------------------------------------------- 3. tabelas
rows = []
unidades = ['pixel', 'município'] if mun_masks else ['pixel']
mun_cache = {}                                   # médias municipais, reaproveitadas no bootstrap
for conj, (obs, preds) in conjuntos.items():
  if mun_masks:
    mun_cache[conj] = (municipal_means(obs), {k: municipal_means(P) for k, P in preds.items()})
  for unidade in unidades:
    o_src = obs if unidade == 'pixel' else mun_cache[conj][0]
    for k, P in preds.items():
      p_src = P if unidade == 'pixel' else mun_cache[conj][1][k]
      for thr, nivel in ALERT_LEVELS.items():
        for s, lead in enumerate(LEADS + ['todos']):
          sel = slice(None) if lead == 'todos' else s
          rows.append({'Conjunto': conj, 'Unidade': unidade, 'Nível de alerta': nivel, 'Modelo': NOMES[k],
                       'Lead Time': 'média +30 a +120' if lead == 'todos' else f'+{lead} min',
                       **{c: round(v, 2) for c, v in classification_scores(
                           *_counts(o_src[:, sel] >= thr, p_src[:, sel] >= thr)).items()}})
df_cls = pd.DataFrame(rows)
cls_csv = os.path.join(OUT_DIR, f'fpluvia_metricas_classificacao_teste_cego_2026_{ENV_TAG}_N{N26}{SUF}.csv')
df_cls.to_csv(cls_csv, index=False)

media = df_cls[df_cls['Lead Time'] == 'média +30 a +120']
cols = ['Unidade', 'Nível de alerta', 'Modelo', 'F1 [%]', 'CSI / IoU [%]', 'Precisão / PPV [%]', 'Recall / POD [%]',
        'FAR [%]', 'Acurácia [%]', 'Acurácia trivial "nunca chove" [%]']
print(f'=== Classificação por nível de alerta — {ROTULO_2026}, média de +30 a +120 min | {DECOD_TXT} | {ENV_LABEL} ===')
print('A acurácia por pixel fica perto de 100% até para quem nunca prevê chuva (coluna "trivial"): '
      'use F1, CSI, POD e FAR para comparar.')
with pd.option_context('display.max_rows', None):
  display(media[media['Conjunto'] == ROTULO_2026][cols].set_index(['Unidade', 'Nível de alerta', 'Modelo']))

if ROTULO_2025:      # F1 por município: 2025 x 2026 lado a lado, com a diferença
  comp = (media[media['Unidade'] == unidades[-1]]
          .pivot_table(index=['Nível de alerta', 'Modelo'], columns='Conjunto', values='F1 [%]', sort=False, dropna=False))
  comp = comp[[ROTULO_2025, ROTULO_2026]]
  comp['Δ F1 (2026 − 2025)'] = comp[ROTULO_2026] - comp[ROTULO_2025]
  comp['Posição em 2026'] = comp.groupby(level=0)[ROTULO_2026].rank(ascending=False, method='min').astype('Int64')
  print(f'\n=== F1 [%] por {unidades[-1]}: 2025 x 2026, média de +30 a +120 min (posição 1 = melhor no nível) ===')
  with pd.option_context('display.max_rows', None):
    display(comp.round(1))


# ------------------------------------------- 4. IC de 95 % do F1 (bootstrap por dias inteiros)
# As janelas de um mesmo dia pertencem ao mesmo evento de chuva e não são independentes: reamostra-se o dia
# inteiro. As mesmas reamostragens valem para todos os métodos (comparação pareada).
def bootstrap_f1(o, preds, ts):
  """o, preds[k]: (N, passos, ...). Devolve {(k, thr): array N_BOOT de F1 [%]} e a estimativa pontual."""
  _, inv = np.unique(pd.to_datetime(ts).normalize().values, return_inverse=True)
  n_dias = inv.max() + 1
  amostras = np.random.default_rng(SEED).integers(0, n_dias, (N_BOOT, n_dias))
  est, pontual = {}, {}
  for thr in ALERT_LEVELS:
    ob = (o >= thr).reshape(len(o), -1)
    for k, P in preds.items():
      pb = (P >= thr).reshape(len(P), -1)
      por_dia = [np.bincount(inv, x.sum(1), n_dias) for x in (ob & pb, ~ob & pb, ob & ~pb)]
      tp, fp, fn = (c[amostras].sum(1) for c in por_dia)
      est[(k, thr)] = 100 * 2 * tp / np.maximum(2 * tp + fp + fn, 1)
      t, f, m = (c.sum() for c in por_dia)
      pontual[(k, thr)] = 100 * 2 * t / max(2 * t + f + m, 1)
  return est, pontual, n_dias


unidade_boot = unidades[-1]
linhas_boot = []
for conj, (obs, preds) in conjuntos.items():
  o_b, p_b = (mun_cache[conj] if unidade_boot == 'município' else (obs, preds))
  est, pontual, n_dias = bootstrap_f1(o_b, p_b, datas[conj])
  for thr, nivel in ALERT_LEVELS.items():
    for k in p_b:
      lo, hi = np.percentile(est[(k, thr)], [2.5, 97.5])
      linha = {'Conjunto': conj, 'Nível de alerta': nivel, 'Modelo': NOMES[k], 'Dias': n_dias,
               'F1 [%]': round(pontual[(k, thr)], 1), 'F1 IC95 [%]': f'{lo:.1f} – {hi:.1f}',
               '_lo': lo, '_hi': hi}
      if k != 'ft':
        d = est[('ft', thr)] - est[(k, thr)]
        dlo, dhi = np.percentile(d, [2.5, 97.5])
        linha['ΔF1 (FT-G − método) IC95'] = f'{dlo:+.1f} – {dhi:+.1f}'
        linha['FT-G em relação ao método'] = ('melhor' if dlo > 0 else 'pior' if dhi < 0 else 'empate técnico')
      linhas_boot.append(linha)
df_boot = pd.DataFrame(linhas_boot)
boot_csv = os.path.join(OUT_DIR, f'fpluvia_f1_{unidade_boot.replace("í", "i")}_bootstrap_dias_2025_2026_{ENV_TAG}{SUF}.csv')
df_boot.drop(columns=['_lo', '_hi']).to_csv(boot_csv, index=False)
print(f'\n=== F1 por {unidade_boot} com IC de 95 % (bootstrap por dias, {N_BOOT} reamostragens), média de +30 a +120 min ===')
print('"melhor"/"pior": o IC da diferença não cruza zero; "empate técnico": cruza zero (diferença não significativa).')
with pd.option_context('display.max_rows', None, 'display.max_colwidth', None):
  display(df_boot.drop(columns=['_lo', '_hi']).fillna('—').set_index(['Conjunto', 'Nível de alerta', 'Modelo']))

# ---------------------------------------------------------------- 5. figuras
styles = {'ft': ('#d62728', 'o', '-'), 'ft_media': ('#9467bd', 'P', '-.'), 'zs': ('#ff7f0e', 'v', '--'),
          'linda': ('#1f77b4', '^', '-'), 'lk': ('#2ca02c', 'd', '--'), 'pers': ('#7f7f7f', 's', ':')}
unidade_plot = unidades[-1]

# 4a. o mesmo painel da Célula 12.1, agora para 2026 (Atenção, por horizonte)
panels = [('Precisão / PPV [%]', 'Precisão (↑)'), ('Recall / POD [%]', 'Recall / POD (↑)'),
          ('F1 [%]', 'F1 (↑)'), ('FAR [%]', 'FAR (↓)')]
fig, axes = plt.subplots(1, 4, figsize=(20, 5.6), constrained_layout=True)
base = df_cls[(df_cls['Conjunto'] == ROTULO_2026) & (df_cls['Unidade'] == unidade_plot) &
              (df_cls['Nível de alerta'] == ALERT_LEVELS[0.5]) & (df_cls['Lead Time'] != 'média +30 a +120')]
for ax, (col, title) in zip(axes, panels):
  for k, nome in NOMES.items():
    sub = base[base['Modelo'] == nome]
    c, mk, ls = styles[k]
    ax.plot(LEADS, sub[col].values, color=c, marker=mk, ls=ls, lw=2, label=nome)
  ax.set_title(title, fontweight='bold')
  ax.set_xlabel('Horizonte (min)')
  ax.set_ylabel('%')
  ax.set_xticks(LEADS)
  ax.grid(True, ls=':', alpha=0.55)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=9)
fig.suptitle(f'Classificação de alerta ≥ 0,5 mm/h por {unidade_plot} — teste cego {ROTULO_2026} | {DECOD_TXT} | {ENV_LABEL}',
             fontsize=12, fontweight='bold')
png_h = os.path.join(OUT_DIR, f'metricas_classificacao_teste_cego_2026_por_horizonte_{ENV_TAG}{SUF}.png')
fig.savefig(png_h, dpi=300, bbox_inches='tight')
plt.show()

# 4b. visão geral: F1 por nível de alerta, 2025 (claro) x 2026 (escuro), um painel por nível
if ROTULO_2025:
  fig, axes = plt.subplots(1, 3, figsize=(20, 5.6), constrained_layout=True, sharey=True)
  nomes = list(NOMES.values())
  CURTO = {'ft': 'F-PluvIA-G' if _esc else 'F-PluvIA-A', 'ft_media': 'F-PluvIA-C8', 'zs': 'Zero-Shot',
           'linda': 'LINDA', 'lk': 'LK', 'pers': 'Persistência'}
  xs = np.arange(len(nomes))
  for ax, nivel in zip(axes, ALERT_LEVELS.values()):
    if nivel not in comp.index.get_level_values(0):
      ax.set_title(f'F1 — {nivel} (sem casos)', fontweight='bold'); continue
    sub = comp.loc[nivel].reindex(nomes)
    cores = [styles[k][0] for k in NOMES]
    def _err(conj):   # barras de erro = IC de 95 % do bootstrap por dias
      g = df_boot[(df_boot['Conjunto'] == conj) & (df_boot['Nível de alerta'] == nivel)].set_index('Modelo').reindex(nomes)
      f1 = sub[conj].values
      return np.vstack([np.clip(f1 - g['_lo'].values, 0, None), np.clip(g['_hi'].values - f1, 0, None)])
    eb = {'ecolor': 'black', 'capsize': 3, 'elinewidth': 0.9}
    b25 = ax.bar(xs - 0.2, sub[ROTULO_2025], 0.4, color=cores, alpha=0.35, edgecolor=cores,
                 yerr=_err(ROTULO_2025) if unidade_boot == unidade_plot else None, error_kw=eb)
    b26 = ax.bar(xs + 0.2, sub[ROTULO_2026], 0.4, color=cores, edgecolor='black', lw=0.6,
                 yerr=_err(ROTULO_2026) if unidade_boot == unidade_plot else None, error_kw=eb)
    for b, v in zip(b26, sub[ROTULO_2026]):
      if np.isfinite(v):
        ax.text(b.get_x() + b.get_width() / 2, v + 3.5, f'{v:.1f}', ha='center', fontsize=8)
    ax.set_title(f'F1 — {nivel}', fontweight='bold')
    ax.set_xticks(xs)
    ax.set_xticklabels([CURTO[k] for k in NOMES], fontsize=9)
    ax.grid(True, axis='y', ls=':', alpha=0.55)
  axes[0].set_ylabel(f'F1 por {unidade_plot} [%] (média +30 a +120 min)')
  fig.legend([plt.Rectangle((0, 0), 1, 1, color='gray', alpha=0.35), plt.Rectangle((0, 0), 1, 1, color='gray', ec='black')],
             [ROTULO_2025 + ' — barra clara', ROTULO_2026 + ' — barra escura'],
             loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
  fig.suptitle(f'Alertas por {unidade_plot}: teste 2025 x teste cego 2026 (barras de erro: IC 95 %, bootstrap por dias) | '
               f'{DECOD_TXT} | {ENV_LABEL}',
               fontsize=12.5, fontweight='bold')
  png_n = os.path.join(OUT_DIR, f'f1_alertas_2025_vs_2026_{ENV_TAG}{SUF}.png')
  fig.savefig(png_n, dpi=300, bbox_inches='tight')
  plt.show()
  print(f'[Célula 13.2] Visão geral -> {png_n}')

CELL13_2_RESULTS = {'classificacao': df_cls, 'f1_2025_vs_2026': comp if ROTULO_2025 else None,
                    'f1_bootstrap': df_boot.drop(columns=['_lo', '_hi'])}
print(f'[Célula 13.2] IC do F1 -> {boot_csv}')
print(f'[Célula 13.2] Tabela -> {cls_csv}\n[Célula 13.2] Painel por horizonte -> {png_h}')
