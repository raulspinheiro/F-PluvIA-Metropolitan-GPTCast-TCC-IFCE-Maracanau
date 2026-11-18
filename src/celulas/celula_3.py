# @title
# Célula 3: Target + Adapter Plurianual (2019–2025) — divisão cronológica estrita por ano,
# pré-alocação contígua em RAM (zero-copy), validação dos lotes .npz e resumo das partições.
#
from abc import ABC, abstractmethod
import datetime
import gc
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

from einops import rearrange
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Protocolo do TCC: treino 2019–2023, validação 2024, teste 2025.
SPLIT_YEARS = {'train': (2019, 2023), 'val': (2024, 2024), 'test': (2025, 2025)}


# 1. TARGET (contrato do padrão GoF Adapter)
class GPTCastDataTarget(ABC):

  @abstractmethod
  def setup(self, stage: Optional[str] = None) -> None: ...

  @abstractmethod
  def train_dataloader(self) -> DataLoader: ...

  @abstractmethod
  def val_dataloader(self) -> DataLoader: ...

  @abstractmethod
  def test_dataloader(self) -> DataLoader: ...

  @abstractmethod
  def get_forecast_input(self, batch: Dict[str, torch.Tensor]) -> np.ma.MaskedArray: ...


# 2. DATASET
class AdaptedMiaradDataset(Dataset):
  """Sequências em dBZ da RMF -> dicionário no formato do GPTCast.

  Treino/validação: janelas de `target_seq_len` quadros (8 = 512 tokens) em [-1, 1].
  Teste: a janela completa de 11 quadros (7 de contexto + 4 de verdade) em [0, 60] dBZ.
  `crop` escolhe onde a janela de 8 quadros é cortada dentro dos 11:
    'random' (treino), 'start' (quadros 0–7) ou 'end' (quadros 3–10, contém os 4 alvos).
  """

  def __init__(
      self,
      sequences_dbz: np.ndarray,
      timestamps: np.ndarray,
      clip_and_normalize: Tuple[float, float, float, float] = (0.0, 60.0, -1.0, 1.0),
      return_single_frame: bool = False,
      target_seq_len: Optional[int] = None,
      random_window_crop: bool = False,
      crop: Optional[str] = None,
  ):
    self.sequences_dbz = sequences_dbz  # view do buffer único: nenhuma cópia
    self.timestamps = np.asarray(timestamps, dtype='datetime64[m]')
    self.min_in, self.max_in, self.min_out, self.max_out = clip_and_normalize
    self.return_single_frame = return_single_frame
    self.target_seq_len = target_seq_len
    self.crop = crop or ('random' if random_window_crop else 'start')
    if self.crop not in ('random', 'start', 'end'):
      raise ValueError(f"crop deve ser 'random', 'start' ou 'end' (recebido: {self.crop}).")

  def __len__(self) -> int:
    if self.return_single_frame:
      return len(self.sequences_dbz) * self.sequences_dbz.shape[1]
    return len(self.sequences_dbz)

  def _normalize(self, x: np.ndarray) -> np.ndarray:
    x = np.clip(x, self.min_in, self.max_in)
    return ((x - self.min_in) / (self.max_in - self.min_in)
            * (self.max_out - self.min_out) + self.min_out).astype(np.float32)

  def __getitem__(self, idx: int) -> Dict[str, object]:
    if self.return_single_frame:  # modo 'vqgan': um quadro por item
      n_f = self.sequences_dbz.shape[1]
      seq_idx, frame_idx = divmod(idx, n_f)
      image_hws = self._normalize(self.sequences_dbz[seq_idx, frame_idx])[:, :, None]
    else:
      seq_idx = idx
      raw = self.sequences_dbz[idx]
      L = self.target_seq_len
      if L is not None and raw.shape[0] > L:
        max_start = raw.shape[0] - L
        start = {'random': np.random.randint(0, max_start + 1), 'start': 0, 'end': max_start}[self.crop]
        raw = raw[start:start + L]
      image_hws = np.transpose(self._normalize(raw), (1, 2, 0))

    h, w, _ = image_hws.shape
    ts = self.timestamps[seq_idx]
    file_code = 'COMP_' + np.datetime_as_string(ts, unit='m').replace('-', '').replace('T', '').replace(':', '')
    return {
        'image': torch.from_numpy(np.ascontiguousarray(image_hws)),
        'mask': torch.zeros((h, w), dtype=torch.bool),
        'file_path_': file_code,
    }


# 3. ADAPTER (GoF Adapter + LightningDataModule)
class IMERGToMiaradAdapter(pl.LightningDataModule, GPTCastDataTarget):
  """Carrega os lotes mensais .npz (Célula 2) num buffer único em RAM e divide por ano.

  A divisão é sempre cronológica por ano (SPLIT_YEARS). Se faltar algum ano, a célula
  para com erro, a menos que `allow_proportional_fallback=True` seja passado de propósito.
  """

  def __init__(
      self,
      adaptee=None,
      cache_dir: Optional[str] = None,
      seq_len: int = 11,
      train_seq_len: int = 8,
      grid_size: int = 128,
      res_km: float = 2.0,
      center_lat: float = -3.80,
      center_lon: float = -38.52,
      train_norm: Tuple[float, float, float, float] = (0.0, 60.0, -1.0, 1.0),
      test_norm: Tuple[float, float, float, float] = (0.0, 60.0, 0.0, 60.0),
      batch_size: int = 16,
      num_workers: Optional[int] = None,
      stage_mode: str = 'gptcast',
      env_tag: Optional[str] = None,
      env_label: Optional[str] = None,
      val_crop: str = 'start',
      allow_proportional_fallback: bool = False,
  ):
    super().__init__()
    # batch_size em self.hparams: exigido por GPTCast.configure_optimizers
    self.save_hyperparameters(ignore=['adaptee'])
    g = globals()
    self.adaptee = adaptee
    self.cache_dir = cache_dir or g.get(
        'SERIES_CACHE_DIR', '/content/drive/MyDrive/GPTCast_RMF/series_2019_2025_2km')
    self.seq_len = int(seq_len)
    self.train_seq_len = int(train_seq_len)
    self.grid_size = int(grid_size)
    self.res_km = float(res_km)
    self.res_deg = self.res_km / 111.12
    self.center_lat = float(center_lat)
    self.center_lon = float(center_lon)
    self.train_norm = tuple(train_norm)
    self.test_norm = tuple(test_norm)
    self.batch_size = int(batch_size)
    self.num_workers = num_workers if num_workers is not None else min(4, os.cpu_count() or 2)
    self.return_single_frame = stage_mode == 'vqgan'
    self.val_crop = val_crop
    self.allow_proportional_fallback = allow_proportional_fallback
    self.env_tag = env_tag or g.get('ENV_TAG', 'Ambiente_Ativo')
    self.env_label = env_label or g.get('ENV_LABEL', 'Hardware Ativo')
    self.results_env_dir = g.get('RESULTS_ENV_DIR', os.path.join(self.cache_dir, f'resultados_{self.env_tag}'))

    self.target_lats, self.target_lons = None, None
    self.all_sequences_dbz, self.all_timestamps = None, None
    self.splits: Dict[str, AdaptedMiaradDataset] = {}
    self.split_summary: Optional[pd.DataFrame] = None
    self.file_report: List[Dict] = []

  # ------------------------------------------------------------- carga e validação
  def _list_files(self) -> List[str]:
    pattern = os.path.join(
        self.cache_dir, f'RMF_*_{int(self.res_km)}km_{self.grid_size}x{self.grid_size}_seq{self.seq_len}.npz')
    files = sorted(glob.glob(pattern))
    if not files:
      raise FileNotFoundError(f'Nenhum lote .npz em {self.cache_dir}. Execute a Célula 2 antes.')
    return files

  def setup(self, stage: Optional[str] = None) -> None:
    if self.all_sequences_dbz is not None and self.splits:
      return
    files = self._list_files()
    self.all_sequences_dbz = self.all_timestamps = None
    gc.collect()

    metas, total, configs = [], 0, set()
    ref_lats = ref_lons = None
    self.file_report = []
    for f in files:
      with np.load(f, allow_pickle=False) as z:
        ts = z['timestamps'].astype('datetime64[m]')
        shape = z['sequences_dbz'].shape
        lats, lons = z['target_lats'], z['target_lons']
        meta = json.loads(str(z['etl_meta'])) if 'etl_meta' in z.files else None
      name = os.path.basename(f)
      if len(shape) != 4 or shape[1:] != (self.seq_len, self.grid_size, self.grid_size) or shape[0] != len(ts):
        raise ValueError(f'{name}: formato {shape} incompatível com {len(ts)} timestamps e a malha esperada.')
      if ref_lats is None:
        ref_lats, ref_lons = lats.copy(), lons.copy()
      elif not (np.allclose(lats, ref_lats, atol=1e-4) and np.allclose(lons, ref_lons, atol=1e-4)):
        raise ValueError(f'{name}: malha geográfica diferente dos demais lotes.')
      if meta is not None:
        configs.add(json.dumps(meta.get('config', {}), sort_keys=True))
      self.file_report.append({'arquivo': name, 'sequencias': len(ts),
                               'metadados': 'v2' if meta else 'legado'})
      if len(ts):
        metas.append((f, len(ts), ts))
        total += len(ts)
    if len(configs) > 1:
      raise ValueError('Lotes .npz gerados com parâmetros diferentes (ex.: stride). Refaça a Célula 2.')
    if total == 0:
      raise ValueError('Os lotes .npz não contêm sequências válidas.')
    self.target_lats, self.target_lons = ref_lats, ref_lons

    # Buffer único pré-alocado (13.893 x 11 x 128 x 128 float32 ≈ 9,3 GB na série completa)
    self.all_sequences_dbz = np.empty((total, self.seq_len, self.grid_size, self.grid_size), np.float32)
    self.all_timestamps = np.empty((total,), 'datetime64[m]')
    cur = 0
    for f, n, ts in metas:
      with np.load(f, allow_pickle=False) as z:
        self.all_sequences_dbz[cur:cur + n] = z['sequences_dbz']
      self.all_timestamps[cur:cur + n] = ts
      cur += n
    del metas
    gc.collect()

    if np.any(np.diff(self.all_timestamps.astype(np.int64)) < 0):
      order = np.argsort(self.all_timestamps, kind='stable')
      self.all_sequences_dbz = self.all_sequences_dbz[order]
      self.all_timestamps = self.all_timestamps[order]
    dup = int(np.sum(np.diff(self.all_timestamps.astype(np.int64)) == 0))
    if dup:
      print(f'[Aviso Célula 3] {dup} timestamps repetidos entre os lotes .npz.')
    n_legacy = sum(r['metadados'] == 'legado' for r in self.file_report)
    print(f'[Célula 3] {len(files)} lotes lidos ({n_legacy} legados, sem metadados da Célula 2 v2).')
    self.create_gptcast_splits()

  # ---------------------------------------------------------------------- divisão
  def create_gptcast_splits(self, val_ratio: float = 0.15, test_ratio: float = 0.15) -> Dict[str, DataLoader]:
    """Divide por ano (SPLIT_YEARS). val_ratio/test_ratio só valem no modo de contingência."""
    if self.all_sequences_dbz is None:
      self.setup()
      return self._loaders()

    years = self.all_timestamps.astype('datetime64[Y]').astype(int) + 1970
    bounds = {}
    for name, (y0, y1) in SPLIT_YEARS.items():
      i0 = int(np.searchsorted(years, y0, side='left'))
      i1 = int(np.searchsorted(years, y1, side='right'))
      bounds[name] = (i0, i1)
    missing = [k for k, (i0, i1) in bounds.items() if i1 <= i0]
    extra = int(np.sum((years < SPLIT_YEARS['train'][0]) | (years > SPLIT_YEARS['test'][1])))

    if missing:
      if not self.allow_proportional_fallback:
        raise ValueError(f'Partições sem dados: {missing}. A série precisa cobrir 2019–2025 '
                         '(ou passe allow_proportional_fallback=True, fora do protocolo do TCC).')
      n = len(years)
      n_te, n_va = max(1, int(n * test_ratio)), max(1, int(n * val_ratio))
      bounds = {'train': (0, n - n_va - n_te), 'val': (n - n_va - n_te, n - n_te), 'test': (n - n_te, n)}
      mode = 'CONTINGÊNCIA proporcional (fora do protocolo do TCC)'
    else:
      mode = 'Cronológico anual (Treino 2019–2023 | Validação 2024 | Teste 2025)'
    if extra:
      print(f'[Aviso Célula 3] {extra} sequências fora de 2019–2025 ficaram de fora das partições.')

    sl = {k: slice(*v) for k, v in bounds.items()}
    S, T = self.all_sequences_dbz, self.all_timestamps
    self.splits = {
        'train': AdaptedMiaradDataset(S[sl['train']], T[sl['train']], self.train_norm,
                                      self.return_single_frame, self.train_seq_len, crop='random'),
        'val': AdaptedMiaradDataset(S[sl['val']], T[sl['val']], self.train_norm,
                                    self.return_single_frame, self.train_seq_len, crop=self.val_crop),
        'test': AdaptedMiaradDataset(S[sl['test']], T[sl['test']], self.test_norm,
                                     False, self.seq_len, crop='start'),
    }
    self._summarize(mode, bounds)
    return self._loaders()

  def _summarize(self, mode: str, bounds: Dict[str, Tuple[int, int]]) -> None:
    total = len(self.all_timestamps)
    rows = []
    for name, rotulo in (('train', 'Treino'), ('val', 'Validação'), ('test', 'Teste')):
      i0, i1 = bounds[name]
      ts = self.all_timestamps[i0:i1]
      n = i1 - i0
      months = sorted({str(t)[:7] for t in ts.astype('datetime64[M]')}) if n else []
      rows.append({
          'Particao': rotulo, 'Sequencias': n, 'Proporcao_pct': round(100 * n / total, 1),
          'Meses': len(months), 'Primeira_janela': str(ts[0]) if n else '', 'Ultima_janela': str(ts[-1]) if n else '',
          'Quadros_por_item': self.seq_len if name == 'test' else self.train_seq_len,
          'Corte_janela': 'completa' if name == 'test' else self.splits[name].crop,
      })
    self.split_summary = pd.DataFrame(rows)
    gb = self.all_sequences_dbz.nbytes / 1024**3
    print(f'[Adapter Split | {self.env_tag} - {mode}]')
    print(f'  -> Total: {total} sequências ({self.res_km} km | buffer único: {gb:.2f} GB) | '
          f'batch {self.batch_size} | workers {self.num_workers}')
    for r in rows:
      print(f"  -> {r['Particao']}: {r['Sequencias']} ({r['Proporcao_pct']}%) | {r['Meses']} meses | "
            f"{r['Primeira_janela']} a {r['Ultima_janela']} | {r['Quadros_por_item']} quadros ({r['Corte_janela']})")
    try:
      os.makedirs(self.results_env_dir, exist_ok=True)
      self.split_summary.to_csv(
          os.path.join(self.results_env_dir, f'resumo_particoes_{self.env_tag}_2019_2025.csv'), index=False)
    except OSError as e:
      print(f'[Aviso Célula 3] Resumo das partições não gravado: {e}')

  # ------------------------------------------------------------------ dataloaders
  def _loader(self, split: str, shuffle: bool) -> DataLoader:
    ds = self.splits[split]
    nw = self.num_workers if (self.num_workers > 0 and len(ds) >= self.batch_size) else 0
    return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, num_workers=nw,
                      pin_memory=torch.cuda.is_available(), persistent_workers=nw > 0,
                      prefetch_factor=2 if nw > 0 else None)

  def _loaders(self) -> Dict[str, DataLoader]:
    return {'train': self.train_dataloader(), 'val': self.val_dataloader(), 'test': self.test_dataloader()}

  def train_dataloader(self) -> DataLoader:
    return self._loader('train', shuffle=True)

  def val_dataloader(self) -> DataLoader:
    return self._loader('val', shuffle=False)

  def test_dataloader(self) -> DataLoader:
    # batch 1 e ordem cronológica: cada item é uma janela de 11 quadros em dBZ
    return DataLoader(self.splits['test'], batch_size=1, shuffle=False, num_workers=1)

  # ----------------------------------------------------------- acesso à inferência
  def get_forecast_input(self, batch: Dict[str, torch.Tensor]) -> np.ma.MaskedArray:
    """Lote (1 item) -> MaskedArray (S, H, W) em dBZ [0, 60], como pede gpt8.forecast()."""
    mask = batch['mask'].cpu().numpy().squeeze()
    data = rearrange(batch['image'].cpu().numpy().squeeze(), 'h w s -> s h w')
    if np.nanmin(data) < -0.01:  # veio normalizado em [-1, 1]
      data = (data + 1.0) * 30.0
    data = np.clip(data, 0.0, 60.0)
    return np.ma.array(data, mask=np.broadcast_to(mask, data.shape))

  def partition_of(self, ts: np.datetime64) -> str:
    y = int(np.datetime64(ts, 'Y').astype(int) + 1970)
    for name, rotulo in (('train', 'treino'), ('val', 'validação'), ('test', 'teste')):
      y0, y1 = SPLIT_YEARS[name]
      if y0 <= y <= y1:
        return rotulo
    return 'fora da série'

  def get_sequence_by_datetime(
      self, target_iso_str: str = '2024-03-18T10:30', tolerance_min: int = 30,
  ) -> Tuple[np.ma.MaskedArray, datetime.datetime]:
    """Janela cujo último quadro é o mais próximo de `target_iso_str`.

    Avisa se a janela encontrada estiver a mais de `tolerance_min` minutos do pedido
    e informa a partição a que pertence (o estudo de caso de 18/03/2024 é de validação).
    """
    if self.all_sequences_dbz is None:
      self.setup()
    target = np.datetime64(target_iso_str, 'm')
    diffs = np.abs((self.all_timestamps - target).astype('timedelta64[m]').astype(np.int64))
    i = int(np.argmin(diffs))
    found = self.all_timestamps[i]
    if diffs[i] > tolerance_min:
      print(f'[Aviso Célula 3] Não há janela terminando em {target_iso_str}; '
            f'usando a mais próxima, {found} ({diffs[i]} min de diferença).')
    print(f'[Célula 3] Janela {found} pertence à partição de {self.partition_of(found)}.')
    seq = np.clip(self.all_sequences_dbz[i], 0.0, 60.0)
    dt = datetime.datetime.strptime(np.datetime_as_string(found, unit='m'), '%Y-%m-%dT%H:%M')
    return np.ma.array(seq, mask=np.zeros_like(seq, dtype=bool)), dt


if 'md_rmf' in globals() and md_rmf is not None:
  md_rmf.cache_dir = globals().get('SERIES_CACHE_DIR', md_rmf.cache_dir)
  md_rmf.env_tag = globals().get('ENV_TAG', md_rmf.env_tag)
  md_rmf.env_label = globals().get('ENV_LABEL', md_rmf.env_label)

print(f'Célula 3 carregada: IMERGToMiaradAdapter (ambiente {globals().get("ENV_TAG", "não definido")}).')
