# @title
# Célula Extra 2: Manifesto de Integridade e Proveniência (SHA-256) — F-PluvIA Metropolitan
# Lista, com hash SHA-256, só os artefatos do projeto (resultados, checkpoints, séries e o notebook),
# registra o código de cada célula, as versões das bibliotecas e o hardware, e grava JSON, Markdown e
# SHA256SUMS em docs/provenance/ (lido pela Célula de Deploy) e no Drive.
import datetime
import fnmatch
import hashlib
import importlib.metadata as md
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

ENV_TAG = globals().get('ENV_TAG', 'Ambiente_Ativo')
ENV_LABEL = globals().get('ENV_LABEL', 'Hardware Ativo')
SERIES_CACHE_DIR = globals().get('SERIES_CACHE_DIR', '/content/drive/MyDrive/GPTCast_RMF/series_2019_2025_2km')
RESULTS_ENV_DIR = globals().get('RESULTS_ENV_DIR', os.path.join(SERIES_CACHE_DIR, f'resultados_{ENV_TAG}'))
CHECKPOINTS_ENV_DIR = globals().get('CHECKPOINTS_ENV_DIR', '')
BASE_DIR = Path(globals().get('BASE_DIR', '/content'))

# ------------------------------------------------------------------- 1. configuração
# Pastas varridas (rótulo -> pasta). Nada fora delas entra no manifesto: o Drive inteiro NUNCA é lido.
PASTAS = {
    'resultados': RESULTS_ENV_DIR,
    'checkpoints': CHECKPOINTS_ENV_DIR,
    'series_2km': SERIES_CACHE_DIR,
    'comparativo_hardware': os.path.join(SERIES_CACHE_DIR, 'comparativo_hardware_G4_vs_H100'),   # Célula 13
}
NOTEBOOK_PATH = globals().get('DETECTED_NB_PATH')        # opcional: .ipynb salvo no Drive
NOTEBOOK_NAME = 'Model_Annual_Full_Training_Series_TCC2_-_2026_2_-_RAUL_SANTIAGO_PINHEIRO.ipynb'
if not (NOTEBOOK_PATH and os.path.isfile(NOTEBOOK_PATH)):          # procura o notebook salvo (Ctrl+S antes)
  for _c in (Path('/content/drive/MyDrive/Colab Notebooks') / NOTEBOOK_NAME, Path('/content/drive/MyDrive') / NOTEBOOK_NAME):
    if _c.is_file():
      NOTEBOOK_PATH = str(_c)
      break
EXTENSOES = {'.csv', '.png', '.pdf', '.json', '.npz', '.ckpt', '.pt', '.pth', '.ipynb', '.html'}
PROFUNDIDADE_MAX = {'series_2km': 0}                      # na pasta das séries, só os arquivos do topo (.npz mensais)
IGNORAR = ('provenance', '.ipynb_checkpoints', '.Trash', 'tmp_gpm_raw', '__pycache__', 'inmet')
IGNORAR_ARQUIVOS = ('*6_orgaos*', '*6orgaos*', '*interinstitucional*', 'inmet_*', '*pluviometros_inmet*')   # antiga Célula 13.1 (órgãos / INMET), removida
INCLUIR_CHECKPOINTS = True                                # .ckpt têm ~1–4 GB; o hash fica em cache
META = {
    # Título: confira com o registrado na coordenação. A comparação com órgãos foi removida do projeto, por isso
    # "e Benchmark Multi-Institucional" saiu daqui.
    'titulo': ('F-PluvIA Metropolitan: Adaptação Cross-Sensor via Fine-Tuning Espaço-Temporal do VQGAN-Transformer '
               'GPTCast-8x8 para Nowcasting Generativo em Séries GPM IMERG-FR na Região Metropolitana de Fortaleza'),
    'formato': 'Trabalho de Conclusão de Curso (TCC) — Artigo Científico',
    'instituicao': 'Instituto Federal de Educação, Ciência e Tecnologia do Ceará (IFCE) — Campus Maracanaú',
    'curso': 'Bacharelado em Ciência da Computação', 'semestre': '2026.2',
    'autor': 'Raul Santiago Pinheiro', 'orientador': 'Amauri Holanda de Souza Júnior',
    'orientador_lattes': 'http://lattes.cnpq.br/2629504708221802',
    'repositorio': 'https://github.com/raulspinheiro/F-PluvIA-Metropolitan-GPTCast-TCC-IFCE-Maracanau',
}

t_ini = time.perf_counter()
agora_utc = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
agora_brt = agora_utc.astimezone(datetime.timezone(datetime.timedelta(hours=-3)))
carimbo = agora_brt.strftime('%Y%m%d_%H%M%S')
DEST = BASE_DIR / 'docs' / 'provenance'
DEST.mkdir(parents=True, exist_ok=True)
DEST_DRIVE = Path(RESULTS_ENV_DIR) / 'provenance'
DEST_DRIVE.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------- 2. hashes (com cache por tamanho e data)
CACHE_HASH = DEST_DRIVE / 'cache_sha256.json'
try:
  _cache = json.loads(CACHE_HASH.read_text()) if CACHE_HASH.exists() else {}
except (OSError, json.JSONDecodeError):
  _cache = {}


def sha256(p: Path) -> str:
  st = p.stat()
  chave = f'{p}|{st.st_size}|{int(st.st_mtime)}'
  if chave in _cache:
    return _cache[chave]
  h = hashlib.sha256()
  with open(p, 'rb') as f:
    for bloco in iter(lambda: f.read(8 << 20), b''):
      h.update(bloco)
  _cache[chave] = h.hexdigest()
  return _cache[chave]


artefatos, falhas = [], []
for rotulo, pasta in PASTAS.items():
  if not pasta or not os.path.isdir(pasta):
    continue
  if rotulo == 'checkpoints' and not INCLUIR_CHECKPOINTS:
    continue
  raiz = Path(pasta)
  for dirpath, dirnames, files in os.walk(raiz):
    rel_dir = Path(dirpath).relative_to(raiz)
    dirnames[:] = sorted(d for d in dirnames if d not in IGNORAR and not d.startswith('.')
                         and len(rel_dir.parts) < PROFUNDIDADE_MAX.get(rotulo, 99))
    for nome in sorted(files):
      p = Path(dirpath) / nome
      if (nome.startswith('.') or p.suffix.lower() not in EXTENSOES or nome == CACHE_HASH.name
          or any(fnmatch.fnmatch(nome, pat) for pat in IGNORAR_ARQUIVOS)):
        continue
      rel = f'{rotulo}/{p.relative_to(raiz).as_posix()}'          # caminho relativo: nada do Drive pessoal
      try:
        artefatos.append({'arquivo': rel, 'bytes': p.stat().st_size, 'sha256': sha256(p)})
      except OSError as e:
        falhas.append({'arquivo': rel, 'erro': f'{type(e).__name__}: {e.strerror or e}'})
if NOTEBOOK_PATH and os.path.isfile(NOTEBOOK_PATH):
  p = Path(NOTEBOOK_PATH)
  artefatos.append({'arquivo': f'notebook/{p.name}', 'bytes': p.stat().st_size, 'sha256': sha256(p)})
artefatos.sort(key=lambda a: a['arquivo'])
try:
  CACHE_HASH.write_text(json.dumps(_cache))
except OSError:
  pass

# Hash raiz: depende só da lista (arquivo, sha256). Qualquer pessoa com os mesmos arquivos chega ao mesmo valor.
raiz_hash = hashlib.sha256(''.join(f"{a['sha256']}  {a['arquivo']}\n" for a in artefatos).encode()).hexdigest()


# ------------------------------------------------------ 3. código das células executadas
def _oculta(s: str) -> str:
  s = re.sub(r'''((?:pass(?:word)?|senha|pwd|token|secret|api_?key)\w*\s*=\s*)(['"]).+?\2''', r'\1\2[OCULTO]\2', s, flags=re.I)
  return re.sub(r'[\w.+-]+@[\w-]+(?:\.[\w-]+)+', '[e-mail oculto]', s)


# Código de todas as células, lido do notebook salvo; sem ele, das células executadas nesta sessão.
codigo, fontes, origem_codigo = {}, [], 'notebook salvo'
if NOTEBOOK_PATH and os.path.isfile(NOTEBOOK_PATH):
  try:
    fontes = [''.join(c.get('source', [])) for c in json.loads(Path(NOTEBOOK_PATH).read_text(encoding='utf-8'))['cells']
              if c.get('cell_type') == 'code']
  except Exception:
    fontes = []
if not fontes:
  ip = get_ipython() if 'get_ipython' in globals() else None
  fontes, origem_codigo = (list(ip.user_ns.get('_ih', [])) if ip is not None else []), 'histórico da sessão'
for i, src in enumerate(fontes):
  if isinstance(src, str) and src.strip():
    m = re.search(r'^#\s*(?:@title\s*#\s*)?(C[ée]lula\s+[^:\n]+)', src, re.M)
    if m:
      codigo[m.group(1).strip()] = {'posicao': i, 'linhas': src.count('\n') + 1,
                                    'sha256_codigo': hashlib.sha256(_oculta(src).encode()).hexdigest()}


# --------------------------------------------------- 4. ambiente, bibliotecas e hardware
def _cmd(c):
  try:
    return subprocess.check_output(c, stderr=subprocess.DEVNULL, text=True, timeout=20).strip()
  except Exception:
    return None


def _versao(pkg):
  try:
    return md.version(pkg)
  except md.PackageNotFoundError:
    return None


libs = {k: _versao(k) for k in ('gptcast', 'gpm-api', 'pysteps', 'torch', 'lightning', 'pytorch-lightning',
                                 'numpy', 'pandas', 'xarray', 'scikit-image', 'cartopy', 'geopandas')}
libs = {k: v for k, v in libs.items() if v}
gpu = {'disponivel': torch.cuda.is_available()}
if torch.cuda.is_available():
  pr = torch.cuda.get_device_properties(0)
  gpu.update({'modelo': pr.name, 'vram_total_gb': round(pr.total_memory / 2 ** 30, 1),
              'pico_vram_alocada_nesta_sessao_gb': round(torch.cuda.max_memory_allocated(0) / 2 ** 30, 2),
              'bf16': bool(torch.cuda.is_bf16_supported()), 'cuda': torch.version.cuda,
              'driver': _cmd(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader']),
              'uuid': _cmd(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'])})

execucao = {}
for nome, var in (('digest_transformer_gpt8', None), ('run_id_celula12', 'RUN_ID'), ('checkpoint', 'best_ckpt_path')):
  if var and var in globals():
    v = globals()[var]
    execucao[nome] = os.path.basename(str(v)) if nome == 'checkpoint' else str(v)
if 'gpt8' in globals():
  h = hashlib.sha256()
  for k, v in sorted(gpt8.state_dict().items()):
    if k.startswith('transformer.'):
      h.update(k.encode())
      h.update(v.detach().float().cpu().numpy().tobytes())
  execucao['digest_transformer_gpt8'] = h.hexdigest()[:12]
reg = os.path.join(CHECKPOINTS_ENV_DIR, f'registro_treino_{ENV_TAG}.json') if CHECKPOINTS_ENV_DIR else ''
if reg and os.path.isfile(reg):
  try:
    execucao['registro_treino'] = json.loads(Path(reg).read_text(encoding='utf-8'))
  except (OSError, json.JSONDecodeError):
    pass

# Caminhos absolutos (ex.: os do registro de treino) viram relativos às pastas do projeto: o manifesto vai para um
# repositório público e não pode listar o Drive pessoal.
_RAIZES = sorted(((str(Path(v)), k) for k, v in PASTAS.items() if v), key=lambda t: -len(t[0]))


def _relativo(x):
  if isinstance(x, dict):
    return {k: _relativo(v) for k, v in x.items()}
  if isinstance(x, (list, tuple)):
    return [_relativo(v) for v in x]
  if isinstance(x, str) and ('/content/' in x or '/MyDrive' in x):
    for raiz, rotulo in _RAIZES:
      if x.startswith(raiz):
        return f'{rotulo}/{os.path.relpath(x, raiz)}'
    return os.path.basename(x.rstrip('/'))
  return x


execucao = _relativo(execucao)
manifesto = {
    'tipo': 'Manifesto de integridade e proveniência (SHA-256)',
    'o_que_comprova': ('Que os arquivos listados não mudaram desde a emissão: basta recalcular o SHA-256 de cada um '
                       '(ou rodar "sha256sum -c" no arquivo SHA256SUMS). Não comprova autoria nem o hardware usado; '
                       'esses campos são declarados pelo ambiente no momento da emissão.'),
    'metadados_academicos': META,
    'emissao': {'utc': agora_utc.isoformat(), 'fortaleza': agora_brt.isoformat(), 'ambiente_tag': ENV_TAG,
                'ambiente': ENV_LABEL, 'python': sys.version.split()[0], 'so': platform.platform()},
    'hardware': gpu,
    'bibliotecas': libs,
    'execucao_do_modelo': execucao,
    'codigo_das_celulas': codigo,
    'artefatos': {'algoritmo': 'SHA-256', 'total': len(artefatos), 'bytes_total': sum(a['bytes'] for a in artefatos),
                  'hash_raiz': raiz_hash, 'pastas': list(k for k, v in PASTAS.items() if v and os.path.isdir(v)),
                  'lista': artefatos, 'falhas_de_leitura': falhas},
}

# ------------------------------------------------------------------ 5. arquivos de saída
gpu_slug = re.sub(r'[^A-Za-z0-9]+', '_', gpu.get('modelo', 'CPU')).strip('_')
base = f'manifesto_proveniencia_{gpu_slug}_{carimbo}'
p_json, p_md, p_sums = DEST / f'{base}.json', DEST / f'{base}.md', DEST / f'SHA256SUMS_{carimbo}.txt'
p_json.write_text(json.dumps(manifesto, indent=2, ensure_ascii=False), encoding='utf-8')
p_sums.write_text(''.join(f"{a['sha256']}  {a['arquivo']}\n" for a in artefatos), encoding='utf-8')

por_pasta = {}
for a in artefatos:
  k = a['arquivo'].split('/', 1)[0]
  n, b = por_pasta.get(k, (0, 0))
  por_pasta[k] = (n + 1, b + a['bytes'])
linhas_md = [
    f"# Manifesto de integridade e proveniência — {META['titulo'].split(':')[0]}", '',
    f"**Emissão:** {agora_brt:%d/%m/%Y %H:%M:%S} (Fortaleza) · {agora_utc:%Y-%m-%d %H:%M:%S} UTC  ",
    f"**Ambiente:** {ENV_LABEL} · GPU {gpu.get('modelo', 'nenhuma')} · Python {sys.version.split()[0]}  ",
    f"**Autor:** {META['autor']} · **Orientador:** {META['orientador']} · {META['instituicao']} · {META['semestre']}", '',
    '## O que este manifesto comprova', '', manifesto['o_que_comprova'], '',
    f"**Hash raiz (SHA-256 da lista de arquivos):** `{raiz_hash}`", '',
    '## Artefatos por pasta', '', '| Pasta | Arquivos | Tamanho |', '|---|---:|---:|',
    *[f'| {k} | {n} | {b / 2 ** 20:,.1f} MB |' for k, (n, b) in sorted(por_pasta.items())], '',
    '## Execução do modelo', '', *[f'- **{k}:** `{v}`' for k, v in execucao.items() if not isinstance(v, dict)], '',
    '## Bibliotecas', '', *[f'- {k} {v}' for k, v in libs.items()], '',
    '## Código das células (SHA-256 do código executado)', '', '| Célula | Linhas | SHA-256 |', '|---|---:|---|',
    *[f"| {k} | {v['linhas']} | `{v['sha256_codigo'][:16]}…` |" for k, v in codigo.items()], '',
    '## Lista completa', '', f'Todos os {len(artefatos)} arquivos e hashes estão em `{p_sums.name}` e `{p_json.name}`.',
    'Para conferir: `sha256sum -c ' + p_sums.name + '` (rodado na pasta que contém resultados/, checkpoints/ e series_2km/).',
]
if falhas:
  linhas_md += ['', f'**{len(falhas)} arquivo(s) não puderam ser lidos** (listados no JSON).']
p_md.write_text('\n'.join(linhas_md) + '\n', encoding='utf-8')

for p in (p_json, p_md, p_sums):
  (DEST_DRIVE / p.name).write_bytes(p.read_bytes())
LAST_JSON_CERT, LAST_MD_CERT, LAST_SHA256SUMS = str(p_json), str(p_md), str(p_sums)

print(f'[Célula Extra 2] {len(artefatos)} artefatos ({sum(a["bytes"] for a in artefatos) / 2 ** 30:.2f} GB) em '
      f'{", ".join(f"{k} ({n})" for k, (n, _) in sorted(por_pasta.items()))}')
print(f'[Célula Extra 2] Código de {len(codigo)} células registrado ({origem_codigo}) | GPU: {gpu.get("modelo", "nenhuma")}')
print(f'[Célula Extra 2] Hash raiz: {raiz_hash}')
if falhas:
  print(f'[Célula Extra 2] Atenção: {len(falhas)} arquivo(s) não lidos (ver JSON).')
print(f'[Célula Extra 2] Arquivos -> {DEST} (para o Deploy) e {DEST_DRIVE} | {time.perf_counter() - t_ini:.1f} s')
