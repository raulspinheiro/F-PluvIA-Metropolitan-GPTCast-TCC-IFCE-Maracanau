# @title
# Célula 1: Preparação do Ambiente, Montagem do Google Drive (2019-2025) e Rastreabilidade por Hardware (G4 vs. H100)
import functools
import glob
import os
import shutil
import subprocess
import sys
from google.colab import drive

# 1. Montagem do Google Drive
drive.mount('/content/drive')

# 2. Definição estrita das pastas dedicadas para a Série Plurianual (2019-2025)
BASE_WORK_DIR = '/content/drive/MyDrive/GPTCast_RMF'
OLD_SERIES_DIR = os.path.join(BASE_WORK_DIR, 'series_2020_2026_2km')
SERIES_CACHE_DIR = os.path.join(BASE_WORK_DIR, 'series_2019_2025_2km')
CHECKPOINTS_DIR = os.path.join(BASE_WORK_DIR, 'checkpoints_rmf_finetuned')
GPM_RAW_TEMP_DIR = '/content/tmp_gpm_raw/GPM'  # SSD local da VM (terminado em /GPM conforme exigido pela gpm-api)

# Migração transparente caso a pasta antiga ainda exista no Drive
if os.path.exists(OLD_SERIES_DIR) and not os.path.exists(SERIES_CACHE_DIR):
  os.rename(OLD_SERIES_DIR, SERIES_CACHE_DIR)

for directory in [
    BASE_WORK_DIR,
    SERIES_CACHE_DIR,
    CHECKPOINTS_DIR,
    GPM_RAW_TEMP_DIR,
]:
  os.makedirs(directory, exist_ok=True)

# 3. Clonagem do repositório oficial GPTCast e inclusão no path de execução
if not os.path.exists('/content/GPTCast'):
  subprocess.run(
      [
          'git',
          'clone',
          'https://github.com/DSIP-FBK/GPTCast.git',
          '/content/GPTCast',
      ],
      check=True,
  )
if '/content/GPTCast' not in sys.path:
  sys.path.append('/content/GPTCast')

# 4. Instalação consolidada de todas as dependências científicas e geoespaciais
try:
  import cartopy
  import geopandas
  import gpm
  import lightning
  import pysteps

  print('[Setup] Bibliotecas científicas e geoespaciais já instaladas na VM.')
except ImportError:
  print(
      '[Setup] A instalar bibliotecas científicas, utilitários geoespaciais e'
      ' Lightning...'
  )
  subprocess.run(
      [
          sys.executable,
          '-m',
          'pip',
          'install',
          '-q',
          'pandas==2.2.3',
          'python-dateutil>=2.8.2,<3',
          'gpm-api',
          'xarray',
          'netCDF4',
          'h5netcdf',
          'scipy',
          'einops',
          'lightning',
          'pytorch-lightning',
          'hydra-core',
          'omegaconf',
          'cartopy',
          'geopandas',
          'pysteps',
      ],
      check=True,
  )

# 5. Desbloqueio do PyTorch 2.6+ para carregar checkpoints do Zenodo/Lightning
import torch

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

if not getattr(torch.load, '_is_patched_for_gptcast', False):
  _orig_torch_load = torch.load

  @functools.wraps(_orig_torch_load)
  def _trusted_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)

  _trusted_torch_load._is_patched_for_gptcast = True
  torch.load = _trusted_torch_load
  torch.serialization.load = _trusted_torch_load

# 6. Deteção de Hardware (G4 Blackwell vs. H100 Hopper vs. CPU) e Isolamento de Artefatos
gpu_available = torch.cuda.is_available()
if gpu_available:
  torch.set_float32_matmul_precision('high')
  torch.backends.cudnn.benchmark = True
  gpu_device_name = torch.cuda.get_device_name(0)
  vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
  bf16_ok = torch.cuda.is_bf16_supported()

  if 'H100' in gpu_device_name.upper():
    ENV_TAG = 'H100_Hopper_80GB'
    ENV_LABEL = f'{gpu_device_name} ({vram_gb:.1f} GB)'
  elif '6000' in gpu_device_name.upper() or 'BLACKWELL' in gpu_device_name.upper():
    ENV_TAG = 'G4_RTX6000_Blackwell_95GB'
    ENV_LABEL = f'{gpu_device_name} ({vram_gb:.1f} GB)'
  elif 'A100' in gpu_device_name.upper():
    ENV_TAG = f'A100_{int(round(vram_gb))}GB'
    ENV_LABEL = f'{gpu_device_name} ({vram_gb:.1f} GB)'
  elif 'L4' in gpu_device_name.upper():
    ENV_TAG = 'L4_Ada_24GB'
    ENV_LABEL = f'{gpu_device_name} ({vram_gb:.1f} GB)'
  else:
    ENV_TAG = (
        gpu_device_name.replace(' ', '_').replace('-', '_')
        + f'_{int(round(vram_gb))}GB'
    )
    ENV_LABEL = f'{gpu_device_name} ({vram_gb:.1f} GB)'
else:
  gpu_device_name = 'Nenhuma GPU detetada (CPU Padrão)'
  vram_gb = 0.0
  bf16_ok = False
  ENV_TAG = 'CPU_Padrao'
  ENV_LABEL = 'CPU Padrão (Google Colab VM)'

# Subpastas isoladas por Ambiente de Execução (evita sobrescrever pesos e tabelas entre G4 e H100)
CHECKPOINTS_ENV_DIR = os.path.join(CHECKPOINTS_DIR, ENV_TAG)
RESULTS_ENV_DIR = os.path.join(SERIES_CACHE_DIR, f'resultados_{ENV_TAG}')
os.makedirs(CHECKPOINTS_ENV_DIR, exist_ok=True)
os.makedirs(RESULTS_ENV_DIR, exist_ok=True)

# Preserva na subpasta da G4 o checkpoint já treinado na raiz de CHECKPOINTS_DIR
if ENV_TAG == 'G4_RTX6000_Blackwell_95GB':
  for ckpt_f in glob.glob(os.path.join(CHECKPOINTS_DIR, '*.ckpt')):
    dst_ckpt = os.path.join(CHECKPOINTS_ENV_DIR, os.path.basename(ckpt_f))
    if not os.path.exists(dst_ckpt):
      shutil.copy2(ckpt_f, dst_ckpt)

print('\n=== Ambiente Model_Annual_Full_Training_Series (2019–2025) Pronto ===')
print(f'Diretório Base: {BASE_WORK_DIR}')
print(f'Cache da Série (2 km): {SERIES_CACHE_DIR}')
print(f'Checkpoints do Ambiente ({ENV_TAG}): {CHECKPOINTS_ENV_DIR}')
print(f'Resultados do Ambiente ({ENV_TAG}): {RESULTS_ENV_DIR}')
print(f'Diretório Temporário Local: {GPM_RAW_TEMP_DIR}')
print(
    f'Acelerador Ativo: {gpu_device_name} | Tag: {ENV_TAG} | VRAM:'
    f' {vram_gb:.1f} GB | Suporte BF16: {bf16_ok}'
)
