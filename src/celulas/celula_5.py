# @title
# Célula 5: Mapas de Precipitação a 2 km sobre os 19 Municípios da RMF — Painel Comparativo
# (Observado x Fine-tuned x Zero-Shot) com Horário de Validade de Cada Quadro.
#
import datetime
import os
import urllib.request
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
from gptcast.utils.converters import dbz_to_rainfall

warnings.filterwarnings('ignore', category=UserWarning)

ENV_TAG = globals().get('ENV_TAG', 'Ambiente_Ativo')
ENV_LABEL = globals().get('ENV_LABEL', 'Hardware Ativo')
RESULTS_ENV_DIR = globals().get('RESULTS_ENV_DIR', os.path.join(SERIES_CACHE_DIR, f'resultados_{ENV_TAG}'))
os.makedirs(RESULTS_ENV_DIR, exist_ok=True)
STEP_MIN = 30
N_LEADS = globals().get('FORECAST_STEPS', 4)

# ---------------------------------------------- 1. limites municipais (19 municípios da RMF)
# Lei Complementar estadual do Ceará; Fortaleza destacada em vermelho.
RMF_MUNICIPIOS = [
    'Aquiraz', 'Cascavel', 'Caucaia', 'Chorozinho', 'Eusébio', 'Fortaleza', 'Guaiúba', 'Horizonte',
    'Itaitinga', 'Maracanaú', 'Maranguape', 'Pacajus', 'Pacatuba', 'Paracuru', 'Paraipaba',
    'Pindoretama', 'São Gonçalo do Amarante', 'São Luís do Curu', 'Trairi',
]
# Malha municipal do IBGE redistribuída em GeoJSON pelo projeto geodata-br (tbrugz/geodata-br).
GEOJSON_CE_URL = 'https://raw.githubusercontent.com/tbrugz/geodata-br/master/geojson/geojs-23-mun.json'
GEOJSON_LOCAL = os.path.join(globals().get('BASE_WORK_DIR', '/content'), 'ceara_municipios_ibge.json')

if not os.path.exists(GEOJSON_LOCAL):
  try:
    urllib.request.urlretrieve(GEOJSON_CE_URL, GEOJSON_LOCAL)
  except Exception as e:
    print(f'[Aviso Célula 5] Malha municipal não baixada ({e}); os mapas sairão sem limites municipais.')

gdf_rmf = gdf_fortaleza = None
if os.path.exists(GEOJSON_LOCAL):
  gdf_ce = gpd.read_file(GEOJSON_LOCAL)
  gdf_rmf = gdf_ce[gdf_ce['name'].isin(RMF_MUNICIPIOS)].copy()
  gdf_fortaleza = gdf_rmf[gdf_rmf['name'] == 'Fortaleza'].copy()
  faltando = sorted(set(RMF_MUNICIPIOS) - set(gdf_rmf['name']))
  print(f'[Célula 5] Limites municipais: {len(gdf_rmf)} de {len(RMF_MUNICIPIOS)} municípios da RMF'
        + (f' | faltando: {", ".join(faltando)}' if faltando else ''))

# ---------------------------------------------------------- 2. malha e escala de cores
lats = md_rmf.target_lats
lons = md_rmf.target_lons
extent_rmf = [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]
lon_grid, lat_grid = np.meshgrid(lons, lats)

rain_levels = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 25.0, 35.0, 50.0, 80.0]   # 10 classes
rain_colors = ['#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#2ca25f',
               '#66c2a4', '#fee08b', '#fdae61', '#f46d43', '#d73027']         # 1 cor por classe
cmap_rain = mcolors.ListedColormap(rain_colors)
cmap_rain.set_under((1, 1, 1, 0))   # abaixo de 0,1 mm/h: transparente
cmap_rain.set_over('#67001f')       # acima de 80 mm/h
norm_rain = mcolors.BoundaryNorm(rain_levels, cmap_rain.N)


def _valid_times(base_dt, n=N_LEADS):
  """base_dt = horário do ÚLTIMO quadro da janela (o +120 min). Devolve (t0, [validades])."""
  if isinstance(base_dt, np.datetime64):
    base_dt = datetime.datetime.strptime(np.datetime_as_string(base_dt, unit='m'), '%Y-%m-%dT%H:%M')
  if not isinstance(base_dt, datetime.datetime):
    return None, [None] * n
  t0 = base_dt - datetime.timedelta(minutes=STEP_MIN * n)
  return t0, [t0 + datetime.timedelta(minutes=STEP_MIN * (i + 1)) for i in range(n)]


def _fmt_t(t):
  if t is None:
    return ''
  brt = t - datetime.timedelta(hours=3)
  return f'{t:%H:%M} UTC ({brt:%H:%M} BRT)'


def _draw_map(ax, field, title, show_left=True):
  ax.set_extent(extent_rmf, crs=ccrs.PlateCarree())
  ax.add_feature(cfeature.LAND, facecolor='#f7f7f7', zorder=0)
  ax.add_feature(cfeature.OCEAN, facecolor='#eef6fa', zorder=0)
  ax.add_feature(cfeature.COASTLINE, linewidth=0.9, edgecolor='#333333', zorder=3)
  if gdf_rmf is not None:
    gdf_rmf.boundary.plot(ax=ax, edgecolor='#555555', linewidth=0.6, zorder=4)
  if gdf_fortaleza is not None:
    gdf_fortaleza.boundary.plot(ax=ax, edgecolor='#d62728', linewidth=1.3, zorder=5)
  f = np.clip(np.nan_to_num(field, nan=0.0), 0.0, 205.0)
  im = ax.pcolormesh(lon_grid, lat_grid, np.ma.masked_less(f, 0.1), cmap=cmap_rain, norm=norm_rain,
                     shading='nearest', transform=ccrs.PlateCarree(), zorder=2)
  ax.set_title(f'{title}\nPico {f.max():.1f} | média {f.mean():.2f} mm/h', fontsize=9.5, fontweight='bold')
  gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.5, linestyle='--')
  gl.top_labels = gl.right_labels = False
  gl.left_labels = show_left
  gl.xlocator = mticker.MultipleLocator(0.6)
  gl.ylocator = mticker.MultipleLocator(0.6)
  gl.xlabel_style = gl.ylabel_style = {'size': 7.5}
  return im


def plot_rmf_spatial(data_4steps, base_dt, panel_title: str, save_filename: str = None):
  """Uma linha de 4 mapas (+30 a +120 min) com o horário de validade de cada quadro."""
  data = np.asarray(data_4steps)
  _, valid = _valid_times(base_dt, data.shape[0])
  fig, axes = plt.subplots(1, data.shape[0], figsize=(4.5 * data.shape[0], 5.0),
                           subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)
  axes = np.atleast_1d(axes)
  for i, ax in enumerate(axes):
    im = _draw_map(ax, data[i], f'+{STEP_MIN * (i + 1)} min · {_fmt_t(valid[i])}', show_left=(i == 0))
  cb = fig.colorbar(im, ax=axes.tolist(), orientation='horizontal', fraction=0.05, pad=0.06, extend='max')
  cb.set_label(f'Precipitação (mm/h) — IMERG reamostrado a {RES_KM} km | {ENV_LABEL}', fontsize=9.5)
  fig.suptitle(panel_title, fontsize=12, fontweight='bold')
  if save_filename:
    path = os.path.join(RESULTS_ENV_DIR, save_filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f'[Célula 5] Mapa salvo -> {path}')
  plt.show()


def plot_rmf_comparison(rows: dict, context_rain, base_dt, panel_title: str, save_filename: str = None):
  """Grade: 1ª coluna = último quadro observado (t0); demais = +30 a +120 min; 1 linha por fonte."""
  t0, valid = _valid_times(base_dt)
  n_rows = len(rows)
  fig, axes = plt.subplots(n_rows, N_LEADS + 1, figsize=(4.0 * (N_LEADS + 1), 4.1 * n_rows),
                           subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)
  axes = np.atleast_2d(axes)
  for r, (label, data) in enumerate(rows.items()):
    if r == 0:
      im = _draw_map(axes[r, 0], context_rain, f't0 observado · {_fmt_t(t0)}')
    else:
      axes[r, 0].axis('off')
      axes[r, 0].text(0.5, 0.5, label, ha='center', va='center', fontsize=12, fontweight='bold',
                      transform=axes[r, 0].transAxes)
    for i in range(N_LEADS):
      im = _draw_map(axes[r, i + 1], data[i], f'{label} · +{STEP_MIN * (i + 1)} min', show_left=False)
  axes[0, 0].text(-0.28, 0.5, list(rows)[0], rotation=90, ha='center', va='center', fontsize=12,
                  fontweight='bold', transform=axes[0, 0].transAxes)
  cb = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.03, pad=0.03, extend='max')
  cb.set_label(f'Precipitação (mm/h) — IMERG reamostrado a {RES_KM} km | validade de +30 a +120 min: '
               f'{_fmt_t(valid[0])} a {_fmt_t(valid[-1])} | {ENV_LABEL}', fontsize=10)
  fig.suptitle(panel_title, fontsize=13, fontweight='bold')
  if save_filename:
    path = os.path.join(RESULTS_ENV_DIR, save_filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f'[Célula 5] Painel salvo -> {path}')
  plt.show()


# --------------------------------------------------------------------- 3. figuras
t0_case, _ = _valid_times(dt)
context_rain_t0 = dbz_to_rainfall(np.clip(np.asarray(input_data_dbz)[-1], 0.0, 60.0))
caso = f'Estudo de caso RMF — t0 = {t0_case:%d/%m/%Y} {_fmt_t(t0_case)} (partição de validação)'

linhas = {'Observado (IMERG-FR v7)': ground_truth_rain, 'GPTCast-8x8 Fine-tuned': forecast8_rain}
if 'forecast8_zeroshot_rain' in globals():
  linhas['GPTCast-8x8 zero-shot'] = forecast8_zeroshot_rain
plot_rmf_comparison(linhas, context_rain_t0, dt, caso, save_filename=f'mapa_00_comparativo_rmf_2km_{ENV_TAG}.png')

# Painéis individuais (mesmos nomes de arquivo usados no backup e no artigo)
plot_rmf_spatial(ground_truth_rain, dt, f'Observado (IMERG-FR v7 a {RES_KM} km) — {caso}',
                 save_filename=f'mapa_01_ground_truth_rmf_2km_{ENV_TAG}.png')
plot_rmf_spatial(forecast8_rain, dt, f'GPTCast-8x8 Fine-tuned na RMF (treino 2019–2023) — {caso}',
                 save_filename=f'mapa_02_gptcast8_finetuned_rmf_2km_{ENV_TAG}.png')
if 'forecast8_zeroshot_rain' in globals():
  plot_rmf_spatial(forecast8_zeroshot_rain, dt, f'GPTCast-8x8 zero-shot (pesos originais) — {caso}',
                   save_filename=f'mapa_03_gptcast8_zeroshot_rmf_2km_{ENV_TAG}.png')
