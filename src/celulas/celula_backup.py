# @title
# Célula Backup: Pacote Completo da Execução (.zip no Drive + download) — notebook e HTML, tabelas e
# figuras com as subpastas preservadas (inclusive as da Célula 13), manifesto SHA-256, código de todas as
# células (lido do notebook salvo) e ambiente Python.
import datetime
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ENV_TAG = globals().get('ENV_TAG', 'Ambiente_Ativo')
ENV_LABEL = globals().get('ENV_LABEL', 'Hardware Ativo')
SERIES_CACHE_DIR = globals().get('SERIES_CACHE_DIR', '/content/drive/MyDrive/GPTCast_RMF/series_2019_2025_2km')
RESULTS_ENV_DIR = Path(globals().get('RESULTS_ENV_DIR', os.path.join(SERIES_CACHE_DIR, f'resultados_{ENV_TAG}')))

# ------------------------------------------------------------------- 1. configuração
NOTEBOOK_NAME = 'Model_Annual_Full_Training_Series_TCC2_-_2026_2_-_RAUL_SANTIAGO_PINHEIRO.ipynb'
NOTEBOOK_PATH = globals().get('DETECTED_NB_PATH')        # se None, procura NOTEBOOK_NAME no Drive
DESTINO = Path(SERIES_CACHE_DIR) / 'backups_completos'
EXT = {'.csv', '.png', '.jpg', '.svg', '.pdf', '.json', '.md', '.txt', '.log', '.html'}
IGNORAR = ['cell12_previsoes_*', 'previsoes', 'cache_*', '*.tmp*', '.ipynb_checkpoints', '__pycache__',
           '*6_orgaos*', '*6orgaos*', '*interinstitucional*', 'inmet_*', '*pluviometros_inmet*']   # antiga Célula 13.1 (órgãos / INMET), removida
# Pastas de resultados fora de RESULTS_ENV_DIR que também entram no pacote (rótulo -> pasta)
PASTAS_EXTRAS = {'comparativo_hardware': Path(SERIES_CACHE_DIR) / 'comparativo_hardware_G4_vs_H100'}   # Célula 13
BAIXAR_NO_NAVEGADOR = True
LIMITE_DOWNLOAD_MB = 1500                                # acima disso o navegador costuma falhar; fica só no Drive

agora = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3)))
carimbo = agora.strftime('%Y%m%d_%H%M%S')
nome = f'BACKUP_FPluvIA_{ENV_TAG}_{carimbo}'
STAGE = Path('/content' if os.path.isdir('/content') else '/tmp') / nome
shutil.rmtree(STAGE, ignore_errors=True)
pastas = {k: STAGE / k for k in ('01_notebook_e_relatorio', '02_tabelas', '03_figuras', '04_proveniencia_e_ambiente',
                                 '05_codigo_das_celulas')}
for p in pastas.values():
  p.mkdir(parents=True)
avisos = []

# ---------------------------------------------------------- 2. notebook e relatório HTML
if not (NOTEBOOK_PATH and os.path.isfile(NOTEBOOK_PATH)):
  for cand in (Path('/content/drive/MyDrive/Colab Notebooks') / NOTEBOOK_NAME, Path('/content/drive/MyDrive') / NOTEBOOK_NAME):
    if cand.is_file():
      NOTEBOOK_PATH = str(cand)
      break
if NOTEBOOK_PATH and os.path.isfile(NOTEBOOK_PATH):
  nb = pastas['01_notebook_e_relatorio'] / Path(NOTEBOOK_PATH).name
  shutil.copy2(NOTEBOOK_PATH, nb)
  salvo = datetime.datetime.fromtimestamp(os.path.getmtime(NOTEBOOK_PATH), agora.tzinfo)
  print(f'[1/5] Notebook copiado (salvo no Drive em {salvo:%d/%m/%Y %H:%M}; se for antigo, use Ctrl+S e rode de novo).')
  r = subprocess.run([sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'html', str(nb),
                      '--output', f'{nb.stem}_completo'], capture_output=True, text=True)
  if r.returncode == 0:
    print('[1/5] Relatório HTML gerado.')
  else:
    avisos.append('HTML não gerado: ' + (r.stderr.strip().splitlines() or ['erro desconhecido'])[-1][:200])
else:
  avisos.append(f'Notebook {NOTEBOOK_NAME} não encontrado no Drive (defina DETECTED_NB_PATH).')

# --------------------------------------------------- 3. tabelas e figuras (subpastas mantidas)
n_tab = n_fig = 0


def copiar_arvore(origem: Path, prefixo: Path = Path()):
  global n_tab, n_fig
  for dirpath, dirnames, files in os.walk(origem):
    dirnames[:] = sorted(d for d in dirnames if not any(fnmatch.fnmatch(d, p) for p in IGNORAR))
    for f in sorted(files):
      src = Path(dirpath) / f
      ext = src.suffix.lower()
      if ext not in EXT or any(fnmatch.fnmatch(f, p) for p in IGNORAR):
        continue
      rel = prefixo / src.relative_to(origem)
      if rel.parts[0] == 'provenance':
        dst = pastas['04_proveniencia_e_ambiente'] / Path(*rel.parts[1:])
      elif ext in {'.png', '.jpg', '.svg', '.pdf'}:
        dst, n_fig = pastas['03_figuras'] / rel, n_fig + 1
      else:
        dst, n_tab = pastas['02_tabelas'] / rel, n_tab + 1
      dst.parent.mkdir(parents=True, exist_ok=True)
      shutil.copy2(src, dst)


if RESULTS_ENV_DIR.is_dir():
  copiar_arvore(RESULTS_ENV_DIR)
else:
  avisos.append(f'Pasta de resultados não encontrada: {RESULTS_ENV_DIR}')
for rotulo, pasta in PASTAS_EXTRAS.items():
  if pasta.is_dir():
    copiar_arvore(pasta, Path(rotulo))
  else:
    avisos.append(f'Pasta {rotulo} não encontrada ({pasta}): rode a célula que a gera, se quiser incluí-la.')
print(f'[2/5] {n_tab} tabelas e {n_fig} figuras copiadas (subpastas preservadas, sem sobrescrever nomes iguais).')

# ------------------------------------------------- 4. código das células e ambiente Python
_EMAIL = re.compile(r'[\w.+-]+@(?!users\.noreply\.github\.com)[\w-]+(?:\.[\w-]+)+')
_SEGREDO = re.compile(r'''((?:pass(?:word)?|senha|pwd|token|secret|api_?key)\w*\s*=\s*)(['"]).+?\2''', re.I)
# Código de TODAS as células, lido do notebook salvo (não só das executadas nesta sessão).
ultima, fontes = {}, []
_nb_copia = next(iter(pastas['01_notebook_e_relatorio'].glob('*.ipynb')), None)
if _nb_copia is not None:
  try:
    fontes = [''.join(c.get('source', [])) for c in json.loads(_nb_copia.read_text(encoding='utf-8'))['cells']
              if c.get('cell_type') == 'code']
  except Exception as e:
    avisos.append(f'Não foi possível ler as células do notebook ({type(e).__name__}); usando o histórico da sessão.')
if not fontes:
  ip = get_ipython() if 'get_ipython' in globals() else None
  fontes = list(ip.user_ns.get('_ih', [])) if ip is not None else []
for k, src in enumerate(fontes, 1):
  m = re.search(r'^#\s*(?:@title\s*#\s*)?C[ée]lula\s+([^:\n]+)', src or '', re.M)
  if m:
    ultima[m.group(1).strip()] = src
  elif _nb_copia is not None and (src or '').strip():
    ultima[f'sem_titulo_{k:02d}'] = src
for rot, src in ultima.items():
  slug = re.sub(r'[^0-9A-Za-z]+', '_', rot.lower().replace('é', 'e')).strip('_')
  txt = _EMAIL.sub('[e-mail oculto]', _SEGREDO.sub(r'\1\2[OCULTO]\2', src))
  (pastas['05_codigo_das_celulas'] / f'celula_{slug}.py').write_text(txt.rstrip() + '\n', encoding='utf-8')

amb = pastas['04_proveniencia_e_ambiente']
with open(amb / 'requirements_ambiente_colab.txt', 'w', encoding='utf-8') as f:
  subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f, stderr=subprocess.DEVNULL, check=False)
smi = shutil.which('nvidia-smi')
if smi:
  (amb / 'nvidia_smi.txt').write_text(subprocess.run([smi], capture_output=True, text=True).stdout, encoding='utf-8')
for var in ('LAST_JSON_CERT', 'LAST_MD_CERT', 'LAST_SHA256SUMS'):   # manifesto desta sessão (Célula Extra 2)
  v = globals().get(var)
  if v and os.path.isfile(v) and not (amb / Path(v).name).exists():
    shutil.copy2(v, amb / Path(v).name)
print(f'[3/5] Código de {len(ultima)} células, pip freeze e manifesto SHA-256 incluídos.')

# ------------------------------------------------ 5. índice do pacote e compactação
conteudo = []
for p in sorted(STAGE.rglob('*')):
  if p.is_file():
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    conteudo.append((p.relative_to(STAGE).as_posix(), p.stat().st_size, h))
(STAGE / 'SHA256SUMS_pacote.txt').write_text(''.join(f'{h}  {r}\n' for r, _, h in conteudo), encoding='utf-8')
(STAGE / 'LEIA-ME.json').write_text(json.dumps({
    'projeto': 'F-PluvIA Metropolitan — TCC, Bacharelado em Ciência da Computação, IFCE Campus Maracanaú (2026.2)',
    'ambiente': ENV_LABEL, 'env_tag': ENV_TAG, 'gerado_em_fortaleza': agora.isoformat(timespec='seconds'),
    'arquivos': len(conteudo), 'tabelas': n_tab, 'figuras': n_fig, 'celulas_com_codigo': sorted(ultima),
    'fora_do_pacote': 'séries .npz, checkpoints .ckpt e blocos de previsões: ficam nas pastas originais do Drive',
    'conferir': 'sha256sum -c SHA256SUMS_pacote.txt (na pasta extraída)', 'avisos': avisos,
}, indent=2, ensure_ascii=False), encoding='utf-8')
print(f'[4/5] Índice SHA-256 de {len(conteudo)} arquivos gravado no pacote.')

DESTINO.mkdir(parents=True, exist_ok=True)
zip_path = DESTINO / f'{nome}.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
  for p in sorted(STAGE.rglob('*')):
    if p.is_file():
      z.write(p, p.relative_to(STAGE).as_posix())
shutil.rmtree(STAGE, ignore_errors=True)
h = hashlib.sha256()
with open(zip_path, 'rb') as f:
  for bloco in iter(lambda: f.read(8 << 20), b''):
    h.update(bloco)
(DESTINO / f'{nome}.zip.sha256').write_text(f'{h.hexdigest()}  {zip_path.name}\n', encoding='utf-8')
mb = zip_path.stat().st_size / 2 ** 20
print(f'[5/5] Pacote salvo no Drive: {zip_path} ({mb:.1f} MB) | SHA-256 {h.hexdigest()[:16]}…')
for a in avisos:
  print(f'  [aviso] {a}')
print('  O notebook e o HTML guardam as saídas como foram exibidas: não publique o pacote sem revisar.')

if BAIXAR_NO_NAVEGADOR:
  if mb > LIMITE_DOWNLOAD_MB:
    print(f'  Download pelo navegador não iniciado (pacote acima de {LIMITE_DOWNLOAD_MB} MB): baixe pelo Drive.')
  else:
    try:
      from google.colab import files
      files.download(str(zip_path))
    except Exception as e:
      print(f'  Download pelo navegador indisponível ({type(e).__name__}); o pacote está no Drive.')
