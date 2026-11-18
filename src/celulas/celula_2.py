# @title
# Célula 2: Adaptee Plurianual IMERG-FR v7 (2019–2025) -> malha de 2 km da RMF, com
# cache validado por metadados, log de ETL sem mistura de leituras de cache e
# relatório de completude dos dados.
#
#
import calendar
import datetime
import json
import os
import shutil
import time
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

try:
  import gpm  # gpm-api
except ImportError:  # permite importar a célula fora do Colab (testes)
  gpm = None

ETL_VERSION = '2.0'
SLOT_MIN = 30  # resolução temporal do IMERG (min)

# Parâmetros que, se mudarem, invalidam um .npz em cache.
_CACHE_KEYS = (
    'product', 'version', 'grid_size', 'res_km', 'center_lat', 'center_lon',
    'seq_len', 'stride', 'wet_mean_thr_mmh', 'wet_max_thr_mmh', 'min_wet_frames',
)

_MASTER_COLS = [
    'Ano_Mes', 'Origem', 'Sequencias_Extraidas', 'Tempo_Download_s',
    'Tempo_Processamento_2km_s', 'Tempo_Total_s', 'Tempo_Total_Formatado',
    'Passos_Esperados', 'Passos_Obtidos', 'Fracao_NaN', 'Metodo_Interp',
    'Stride', 'Data_Processamento_UTC', 'Ambiente_Tag', 'Hardware_Ativo',
]


class IMERGExtractorAdaptee:
  """Adaptee (padrão GoF Adapter) da série IMERG-FR para o formato do GPTCast-8x8.

  Para cada mês: baixa os grânulos IMERG-FR v7 (0,1°, 30 min) via gpm-api,
  recorta a RMF com margem de 0,5°, interpola para 128 x 128 pixels de 2 km,
  converte para dBZ (Z = 200 R^1,6, faixa 0–60 dBZ), seleciona janelas de
  `seq_len` quadros consecutivos com chuva e grava um .npz por mês no Drive.

  Observações:
    * a malha de 2 km é reamostragem de um produto de ~11 km: não acrescenta
      detalhe físico; a spline cúbica pode criar picos acima do original
      (valores negativos são zerados);
    * as janelas não atravessam a virada do mês;
    * o passo entre janelas é `stride` x 30 min (stride=2 -> 60 min).
  """

  def __init__(
      self,
      username_pps: Optional[str] = None,
      password_pps: Optional[str] = None,
      raw_temp_dir: Optional[str] = None,
      cache_dir: Optional[str] = None,
      results_env_dir: Optional[str] = None,
      env_tag: Optional[str] = None,
      env_label: Optional[str] = None,
      product: str = 'IMERG-FR',
      version: int = 7,
      grid_size: int = 128,
      res_km: float = 2.0,
      center_lat: float = -3.80,
      center_lon: float = -38.52,
      seq_len: int = 11,
      wet_mean_thr_mmh: float = 0.02,
      wet_max_thr_mmh: float = 1.0,
      strict_cache: bool = False,
      sanitize_master_log: bool = True,
  ):
    g = globals()
    raw_temp_dir = raw_temp_dir or g.get('GPM_RAW_TEMP_DIR', '/content/tmp_gpm_raw/GPM')
    cache_dir = cache_dir or g.get(
        'SERIES_CACHE_DIR', '/content/drive/MyDrive/GPTCast_RMF/series_2019_2025_2km')
    # A gpm-api exige que o diretório base termine em '/GPM'.
    if os.path.basename(os.path.normpath(raw_temp_dir)) != 'GPM':
      raw_temp_dir = os.path.join(raw_temp_dir, 'GPM')

    self.raw_temp_dir = raw_temp_dir
    self.cache_dir = cache_dir
    self.env_tag = env_tag or g.get('ENV_TAG', 'Ambiente_Ativo')
    self.env_label = env_label or g.get('ENV_LABEL', 'Hardware Ativo')
    self.results_env_dir = results_env_dir or g.get(
        'RESULTS_ENV_DIR', os.path.join(cache_dir, f'resultados_{self.env_tag}'))

    self.product = product
    self.version = int(version)
    self.grid_size = int(grid_size)
    self.res_km = float(res_km)
    self.res_deg = self.res_km / 111.12
    self.center_lat = float(center_lat)
    self.center_lon = float(center_lon)
    self.seq_len = int(seq_len)
    self.wet_mean_thr = float(wet_mean_thr_mmh)
    self.wet_max_thr = float(wet_max_thr_mmh)
    self.min_wet_frames = max(3, self.seq_len // 2)
    self.strict_cache = bool(strict_cache)
    self.session_id = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:6]
    self.session_records: List[Dict] = []

    for d in (self.raw_temp_dir, self.cache_dir, self.results_env_dir):
      os.makedirs(d, exist_ok=True)

    self.log_csv_path = os.path.join(self.cache_dir, 'log_tempos_etl_imerg_2019_2025.csv')
    old_csv = os.path.join(self.cache_dir, 'log_tempos_etl_imerg_2020_2026.csv')
    if os.path.exists(old_csv) and not os.path.exists(self.log_csv_path):
      os.rename(old_csv, self.log_csv_path)
    self.env_log_csv_path = os.path.join(
        self.results_env_dir, f'log_tempos_etl_sessao_{self.env_tag}_2019_2025.csv')
    if sanitize_master_log:
      self._sanitize_master_log()

    # Credenciais: argumentos ou variáveis de ambiente (nunca impressas).
    # Na NASA PPS o usuário e a senha são o e-mail cadastrado.
    self._user = username_pps or os.environ.get('PPS_USERNAME')
    self._pwd = password_pps or os.environ.get('PPS_PASSWORD') or self._user
    self._configured = False

    # Malha alvo: 128 x 128 pixels de 2 km (256 x 256 km) centrada na RMF.
    half = (self.grid_size * self.res_deg) / 2.0
    self.target_lats = np.linspace(
        self.center_lat + half, self.center_lat - half, self.grid_size).astype(np.float32)
    self.target_lons = np.linspace(
        self.center_lon - half, self.center_lon + half, self.grid_size).astype(np.float32)
    # Margem de 0,5° evita efeito de borda na spline.
    self.raw_extent = [
        float(self.target_lons.min() - 0.5), float(self.target_lons.max() + 0.5),
        float(self.target_lats.min() - 0.5), float(self.target_lats.max() + 0.5),
    ]

  # ------------------------------------------------------------------ utilitários
  @staticmethod
  def _fmt(seconds: float) -> str:
    if seconds is None or not np.isfinite(seconds):
      return 'não medido'
    m, s = divmod(float(seconds), 60)
    h, m = divmod(int(m), 60)
    return f'{h:02d}h:{m:02d}m:{s:05.2f}s'

  @staticmethod
  def rainfall_to_dbz(rain_rate: np.ndarray) -> np.ndarray:
    """Marshall-Palmer inversa (Z = 200 R^1,6), R limitado a 0–205 mm/h, saída 0–60 dBZ."""
    rain = np.clip(np.nan_to_num(rain_rate, nan=0.0), 0.0, 205.0)
    dbz = np.zeros_like(rain, dtype=np.float32)
    wet = rain > 0.0365  # abaixo disso dBZ seria negativo
    dbz[wet] = 10.0 * np.log10(200.0 * np.power(rain[wet], 1.6))
    return np.round(np.clip(dbz, 0.0, 60.0), 1).astype(np.float32)

  def _config(self, stride: int) -> Dict:
    return {
        'product': self.product, 'version': self.version, 'grid_size': self.grid_size,
        'res_km': self.res_km, 'center_lat': self.center_lat, 'center_lon': self.center_lon,
        'seq_len': self.seq_len, 'stride': int(stride), 'wet_mean_thr_mmh': self.wet_mean_thr,
        'wet_max_thr_mmh': self.wet_max_thr, 'min_wet_frames': self.min_wet_frames,
    }

  def _ensure_gpm_configured(self) -> None:
    if self._configured:
      return
    if gpm is None:
      raise ImportError('gpm-api não está instalada (Célula 1).')
    if not self._user:
      raise ValueError('Credencial PPS ausente: passe username_pps ou defina PPS_USERNAME.')
    gpm.define_configs(
        username_pps=self._user, password_pps=self._pwd,
        username_earthdata=self._user, password_earthdata=self._pwd,
        base_dir=self.raw_temp_dir,
    )
    self._configured = True

  # ------------------------------------------------------------- transformação
  def _interpolate_to_high_res_grid(self, ds: xr.Dataset) -> Tuple[xr.DataArray, str, float]:
    """Interpola ~11 km -> 2 km por spline cúbica separável. Devolve (campo, método, fração NaN)."""
    da = ds['precipitation'].transpose('time', 'lat', 'lon')
    nan_frac = float(np.isnan(da.values).mean()) if da.size else 0.0
    da = da.fillna(0.0)
    method = 'cubic'
    try:
      out = da.interp(lat=self.target_lats, method='cubic').interp(
          lon=self.target_lons, method='cubic')
    except Exception as e:  # registrado, nunca silencioso
      method = 'linear'
      print(f'[Aviso Célula 2] Spline cúbica falhou ({type(e).__name__}: {e}); usando interpolação linear.')
      out = da.interp(lat=self.target_lats, lon=self.target_lons, method='linear')
    out = out.clip(min=0.0, max=205.0).fillna(0.0).astype(np.float32)
    return out, method, nan_frac

  def _extract_valid_sequences(
      self, da_rain: xr.DataArray, stride: int = 1
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Janelas de seq_len quadros contínuos (30 min) com >= min_wet_frames quadros úmidos.

    Quadro úmido: média espacial >= wet_mean_thr e pico >= wet_max_thr (mm/h).
    O timestamp gravado é o do último quadro da janela.
    """
    rain = da_rain.values
    times = da_rain['time'].values.astype('datetime64[m]')
    empty = (np.empty((0, self.seq_len, self.grid_size, self.grid_size), np.float32),
             np.empty((0,), 'datetime64[m]'))
    if rain.shape[0] < self.seq_len:
      return empty
    is_wet = (rain.mean(axis=(1, 2)) >= self.wet_mean_thr) & (rain.max(axis=(1, 2)) >= self.wet_max_thr)
    step_ok = np.diff(times) == np.timedelta64(SLOT_MIN, 'm')

    seqs, v_times = [], []
    for i in range(0, len(times) - self.seq_len + 1, max(1, int(stride))):
      if is_wet[i:i + self.seq_len].sum() >= self.min_wet_frames and step_ok[i:i + self.seq_len - 1].all():
        seqs.append(self.rainfall_to_dbz(rain[i:i + self.seq_len]))
        v_times.append(times[i + self.seq_len - 1])
    if not seqs:
      return empty
    return np.stack(seqs).astype(np.float32), np.array(v_times, dtype='datetime64[m]')

  # ------------------------------------------------------------------- cache
  def _cache_path(self, year: int, month: int) -> str:
    tag = f'RMF_{year}_{month:02d}_{int(self.res_km)}km_{self.grid_size}x{self.grid_size}_seq{self.seq_len}.npz'
    return os.path.join(self.cache_dir, tag)

  def _check_cache(self, npz_path: str, stride: int) -> Tuple[bool, int, str]:
    """(utilizável, nº de sequências, situação). Situação: 'valido', 'legado' ou o motivo da recusa."""
    with np.load(npz_path, allow_pickle=False) as z:
      n = int(len(z['timestamps']))
      shape = z['sequences_dbz'].shape
      meta = json.loads(str(z['etl_meta'])) if 'etl_meta' in z.files else None
    if len(shape) != 4 or shape[1:] != (self.seq_len, self.grid_size, self.grid_size):
      return False, n, f'formato {shape} incompatível'
    if meta is None:
      # .npz anterior à versão 2.0: sem metadados para conferir.
      return (not self.strict_cache), n, 'legado'
    want = self._config(stride)
    diff = [k for k in _CACHE_KEYS if meta.get('config', {}).get(k) != want[k]]
    if diff:
      return False, n, 'parâmetros diferentes: ' + ', '.join(diff)
    return True, n, 'valido'

  # --------------------------------------------------------------------- logs
  def _sanitize_master_log(self) -> None:
    """Leituras de cache não são tempo de ETL: zera (NaN) esses tempos no CSV mestre.

    Uma cópia do arquivo original é guardada antes da primeira alteração.
    """
    if not os.path.exists(self.log_csv_path):
      return
    df = pd.read_csv(self.log_csv_path)
    if 'Origem' not in df.columns:
      return
    mask = df['Origem'].astype(str).str.startswith('Cache') & df['Tempo_Total_s'].notna()
    if not mask.any():
      return
    backup = self.log_csv_path.replace('.csv', '_antes_etl_v2.csv')
    if not os.path.exists(backup):
      shutil.copy2(self.log_csv_path, backup)
    for c in ('Tempo_Download_s', 'Tempo_Processamento_2km_s', 'Tempo_Total_s'):
      if c in df.columns:
        df.loc[mask, c] = np.nan
    df.loc[mask, 'Tempo_Total_Formatado'] = 'não medido'
    df.loc[mask, 'Origem'] = 'Cache Drive (tempo de ETL não registrado)'
    df.to_csv(self.log_csv_path, index=False)
    print(f'[Célula 2] Log mestre saneado: {int(mask.sum())} meses com tempo de leitura de cache '
          f'passaram a "não medido" (cópia em {os.path.basename(backup)}).')

  def _upsert_master(self, rec: Dict) -> None:
    """CSV mestre = uma linha por mês com o tempo REAL de download + processamento.

    Uma leitura de cache só cria linha se o mês ainda não constar, e com tempos NaN.
    """
    df = pd.read_csv(self.log_csv_path) if os.path.exists(self.log_csv_path) else pd.DataFrame(columns=_MASTER_COLS)
    exists = rec['Ano_Mes'] in set(df.get('Ano_Mes', pd.Series(dtype=str)).astype(str))
    is_cache = str(rec['Origem']).startswith('Cache')
    if exists and is_cache:
      return
    if exists:
      df = df[df['Ano_Mes'].astype(str) != rec['Ano_Mes']]
    df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
    df = df.sort_values('Ano_Mes').reset_index(drop=True)
    df.to_csv(self.log_csv_path, index=False)

  def _append_session(self, rec: Dict) -> None:
    """CSV do ambiente = o que aconteceu NESTA sessão, mês a mês (cache ou download)."""
    row = dict(rec, Sessao_ID=self.session_id)
    self.session_records.append(row)
    df_new = pd.DataFrame([row])
    if os.path.exists(self.env_log_csv_path):
      df_old = pd.read_csv(self.env_log_csv_path)
      if 'Sessao_ID' not in df_old.columns:  # arquivo antigo: recomeça com o novo formato
        df_old = pd.DataFrame()
      df_new = pd.concat([df_old, df_new], ignore_index=True)
    df_new.to_csv(self.env_log_csv_path, index=False)

  def _record(self, **kw) -> Dict:
    rec = {c: np.nan for c in _MASTER_COLS}
    rec.update(kw)
    rec['Tempo_Total_Formatado'] = self._fmt(rec.get('Tempo_Total_s'))
    rec['Ambiente_Tag'] = self.env_tag
    rec['Hardware_Ativo'] = self.env_label
    rec['Data_Processamento_UTC'] = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    return rec

  # ------------------------------------------------------------------- etapas
  def fetch_and_process_month(
      self, year: int, month: int, stride: int = 1,
      force_reprocess: bool = False, purge_raw: bool = True,
  ) -> str:
    """Download -> recorte -> 2 km -> dBZ -> janelas -> .npz -> limpeza -> logs, para um mês."""
    npz_path = self._cache_path(year, month)
    ym = f'{year}-{month:02d}'

    if os.path.exists(npz_path) and not force_reprocess:
      t0 = time.perf_counter()
      ok, n_seqs, status = self._check_cache(npz_path, stride)
      t_read = time.perf_counter() - t0
      if ok:
        print(f'[Célula 2 | cache {status}] {ym}: {n_seqs} sequências ({t_read:.2f} s de leitura)')
        self._upsert_master(self._record(
            Ano_Mes=ym, Origem='Cache Drive (tempo de ETL não registrado)',
            Sequencias_Extraidas=n_seqs, Stride=stride))
        self._append_session(self._record(
            Ano_Mes=ym, Origem=f'Cache Drive ({status})', Sequencias_Extraidas=n_seqs,
            Tempo_Download_s=0.0, Tempo_Processamento_2km_s=round(t_read, 2),
            Tempo_Total_s=round(t_read, 2), Stride=stride))
        return npz_path
      print(f'[Célula 2] Cache de {ym} recusado ({status}); reprocessando.')

    self._ensure_gpm_configured()
    last_day = calendar.monthrange(year, month)[1]
    start = datetime.datetime(year, month, 1, 0, 0)
    end = datetime.datetime(year, month, last_day, 23, 59)
    expected = last_day * 24 * 60 // SLOT_MIN

    t0 = time.perf_counter()
    print(f'\n[Célula 2] {ym}: baixando {self.product} v{self.version} ({start:%d/%m} a {end:%d/%m})...')
    gpm.download(product=self.product, product_type='RS', version=self.version,
                 start_time=start, end_time=end, force_download=False, n_threads=4)
    t_dl = time.perf_counter() - t0

    t1 = time.perf_counter()
    ds = gpm.open_dataset(product=self.product, product_type='RS', version=self.version,
                          start_time=start, end_time=end, variables=['precipitation'])
    try:
      ds_crop = ds.gpm.crop(extent=self.raw_extent).load()
      obtained = int(ds_crop.sizes.get('time', 0))
      da2, method, nan_frac = self._interpolate_to_high_res_grid(ds_crop)
      seqs, times = self._extract_valid_sequences(da2, stride=stride)
    finally:
      ds.close()

    meta = {
        'etl_version': ETL_VERSION, 'config': self._config(stride), 'interp_method': method,
        'steps_expected': expected, 'steps_obtained': obtained, 'nan_fraction': round(nan_frac, 6),
        'created_utc': datetime.datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'env_tag': self.env_tag,
    }
    tmp_path = npz_path + '.tmp.npz'  # escrita atômica: nunca deixa .npz corrompido no Drive
    np.savez_compressed(tmp_path, sequences_dbz=seqs, timestamps=times,
                        target_lats=self.target_lats, target_lons=self.target_lons,
                        etl_meta=np.array(json.dumps(meta)))
    os.replace(tmp_path, npz_path)
    del ds_crop, da2

    if purge_raw:
      shutil.rmtree(self.raw_temp_dir, ignore_errors=True)
      os.makedirs(self.raw_temp_dir, exist_ok=True)

    t_proc = time.perf_counter() - t1
    t_tot = time.perf_counter() - t0
    rec = self._record(
        Ano_Mes=ym, Origem='Download NASA + Interp 2km', Sequencias_Extraidas=int(len(seqs)),
        Tempo_Download_s=round(t_dl, 2), Tempo_Processamento_2km_s=round(t_proc, 2),
        Tempo_Total_s=round(t_tot, 2), Passos_Esperados=expected, Passos_Obtidos=obtained,
        Fracao_NaN=round(nan_frac, 6), Metodo_Interp=method, Stride=stride)
    self._upsert_master(rec)
    self._append_session(rec)

    alerta = '' if obtained == expected else f' | ATENÇÃO: {obtained}/{expected} passos de 30 min'
    print(f'[Célula 2] {ym}: {len(seqs)} sequências | download {self._fmt(t_dl)} | '
          f'processamento {self._fmt(t_proc)} | interp {method} | NaN {nan_frac:.2%}{alerta}')
    return npz_path

  def build_period_series(
      self, start_ym: Tuple[int, int] = (2019, 2), end_ym: Tuple[int, int] = (2025, 5),
      months_filter: Optional[List[int]] = None, stride: int = 1,
  ) -> List[str]:
    """Percorre o período mês a mês e imprime um resumo que separa sessão e histórico."""
    files, falhas = [], []
    t0 = time.perf_counter()
    y, m = start_ym
    while (y, m) <= end_ym:
      if months_filter is None or m in months_filter:
        try:
          files.append(self.fetch_and_process_month(y, m, stride=stride, purge_raw=True))
        except Exception as e:
          falhas.append(f'{y}-{m:02d}')
          print(f'[Erro Célula 2] {y}-{m:02d}: {type(e).__name__}: {e}')
      m, y = (1, y + 1) if m == 12 else (m + 1, y)
    t_sess = time.perf_counter() - t0

    sess = pd.DataFrame(self.session_records)
    n_cache = int(sess['Origem'].astype(str).str.startswith('Cache').sum()) if len(sess) else 0
    n_dl = len(sess) - n_cache
    n_seqs = int(sess['Sequencias_Extraidas'].sum()) if len(sess) else 0
    hist = pd.read_csv(self.log_csv_path) if os.path.exists(self.log_csv_path) else pd.DataFrame()
    medido = hist['Tempo_Total_s'].dropna() if 'Tempo_Total_s' in hist else pd.Series(dtype=float)

    print(f'\n=== Resumo da Célula 2 ({self.env_label}) ===')
    print(f'Meses nesta sessão: {len(files)} ({n_dl} baixados, {n_cache} lidos do cache) | '
          f'{n_seqs} sequências | passo entre janelas: {stride * SLOT_MIN} min')
    print(f'Tempo desta sessão: {self._fmt(t_sess)}')
    print(f'Tempo de ETL medido no histórico: {self._fmt(medido.sum()) if len(medido) else "não medido"} '
          f'em {len(medido)} de {len(hist)} meses; os demais não têm tempo registrado.')
    if falhas:
      print(f'Meses com falha: {", ".join(falhas)}')
    print(f'Log mestre: {self.log_csv_path}\nLog da sessão: {self.env_log_csv_path}')
    return files


print(f'Célula 2 carregada: IMERGExtractorAdaptee v{ETL_VERSION} '
      f'(ambiente {globals().get("ENV_TAG", "não definido")}).')
