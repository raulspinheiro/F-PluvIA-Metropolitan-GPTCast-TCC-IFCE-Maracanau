# @title
# Célula Deploy: Envio dos Artefatos para o GitHub — tabelas e figuras de RESULTS_ENV_DIR, manifesto
# SHA-256 (Célula Extra 2), código das células e o notebook.
#
import base64
import datetime
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ENV_TAG = globals().get('ENV_TAG', 'Ambiente_Ativo')
RESULTS_ENV_DIR = Path(globals().get('RESULTS_ENV_DIR', os.path.join(globals().get('SERIES_CACHE_DIR', '/content'),
                                                                     f'resultados_{ENV_TAG}')))
BASE_DIR = Path(globals().get('BASE_DIR', '/content'))

# ------------------------------------------------------------------- 1. configuração
GITHUB_USER = 'raulspinheiro'
REPO_NAME = 'F-PluvIA-Metropolitan-GPTCast-TCC-IFCE-Maracanau'
REPO_URL = globals().get('REPO_URL', f'https://github.com/{GITHUB_USER}/{REPO_NAME}.git')
GIT_USER_NAME = 'Raul Santiago Pinheiro'
GIT_USER_EMAIL = 'raulspinheiro@users.noreply.github.com'
NOTEBOOK_NAME = 'Model_Annual_Full_Training_Series_TCC2_-_2026_2_-_RAUL_SANTIAGO_PINHEIRO.ipynb'
NOTEBOOK_PATH = globals().get('DETECTED_NB_PATH')        # se None, procura NOTEBOOK_NAME no Drive
CLONE_DIR = Path('/content/github_repo_sync') if os.path.isdir('/content') else BASE_DIR / 'github_repo_sync'
MAX_MB = 95.0                                            # o GitHub recusa arquivos acima de 100 MB
EXT_RESULTADOS = {'.csv', '.png', '.pdf', '.json', '.md', '.txt'}
IGNORAR_RESULTADOS = ['cell12_previsoes_*', 'previsoes', 'provenance', 'cache_*', '*.tmp*', '.ipynb_checkpoints',
                      # saídas da antiga Célula 13.1 (benchmark dos 6 órgãos e validação INMET), removida do projeto
                      '*6_orgaos*', 'inmet_*', '*pluviometros_inmet*']
EXPORTAR_CELULAS = True                                  # grava o código de cada célula em src/celulas/*.py
REMOVER_CERTIFICADOS_COM_CAMINHOS = True                 # git rm dos certificados antigos que listam o Drive
CRIAR_README_SE_FALTAR = True                            # cria um README.md só se o repositório não tiver
SO_PREPARAR = False                                      # True = monta o commit e mostra a lista, sem push

# ---------------------------------------------------------------- 2. credencial
try:
  from google.colab import userdata
  GITHUB_TOKEN = (userdata.get('GITHUB_TOKEN') or '').strip()
except Exception:
  GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '').strip()
if not GITHUB_TOKEN and REPO_URL.startswith('https://'):
  raise RuntimeError("Defina 'GITHUB_TOKEN' nos Segredos do Colab (ícone de chave) com 'Notebook access' ativado.")

# O token vai só no cabeçalho HTTP de cada comando; nunca fica no .git/config nem na URL.
_AUTH = []
if GITHUB_TOKEN:
  _b64 = base64.b64encode(f'x-access-token:{GITHUB_TOKEN}'.encode()).decode()
  _AUTH = ['-c', f'http.https://github.com/.extraheader=AUTHORIZATION: basic {_b64}']


def git(*args, cwd=CLONE_DIR, auth=False) -> str:
  r = subprocess.run(['git', *(_AUTH if auth else []), *args], cwd=cwd, text=True, capture_output=True)
  if r.returncode != 0:
    msg = (r.stderr + r.stdout)
    for s in filter(None, (GITHUB_TOKEN, _AUTH[1] if _AUTH else '')):
      msg = msg.replace(s, '***')
    raise RuntimeError(f'git {args[0]} falhou:\n{msg.strip()}')
  return r.stdout.strip()


# ------------------------------------------------------------- 3. clone limpo
if CLONE_DIR.exists():
  shutil.rmtree(CLONE_DIR)
print(f'[1/5] Clonando {REPO_URL} ...')
git('clone', '--depth', '1', REPO_URL, str(CLONE_DIR), cwd=None, auth=True)
git('config', 'user.name', GIT_USER_NAME)
git('config', 'user.email', GIT_USER_EMAIL)
REPO_VAZIO = not git('ls-remote', '--heads', 'origin', auth=True)
if REPO_VAZIO:                       # repositório recém-criado, sem nenhum commit
  RAMO = 'main'
  git('checkout', '-q', '-b', RAMO)
  print('  Repositório vazio: o primeiro commit criará o ramo main.')
else:
  RAMO = git('rev-parse', '--abbrev-ref', 'HEAD')

copiados, grandes = [], []


def copiar(src: Path, dst_rel: str):
  mb = src.stat().st_size / 2 ** 20
  if mb > MAX_MB:
    grandes.append((str(src.name), round(mb, 1)))
    return
  dst = CLONE_DIR / dst_rel
  dst.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(src, dst)
  copiados.append(dst_rel)


# ------------------------------------------------------------ 4. o que vai para o repositório
print('[2/5] Copiando artefatos ...')
# 4a. Tabelas e figuras geradas pelas células (ficam no Drive, em RESULTS_ENV_DIR)
if RESULTS_ENV_DIR.is_dir():
  for dirpath, dirnames, files in os.walk(RESULTS_ENV_DIR):
    dirnames[:] = sorted(d for d in dirnames if not any(fnmatch.fnmatch(d, p) for p in IGNORAR_RESULTADOS))
    for nome in sorted(files):
      p = Path(dirpath) / nome
      if p.suffix.lower() in EXT_RESULTADOS and not any(fnmatch.fnmatch(nome, pat) for pat in IGNORAR_RESULTADOS):
        sub = 'figures' if p.suffix.lower() in {'.png', '.pdf'} else 'tables'
        copiar(p, f'results/{ENV_TAG}/{sub}/{p.relative_to(RESULTS_ENV_DIR).as_posix()}')
else:
  print(f'  [aviso] {RESULTS_ENV_DIR} não existe: nenhuma tabela ou figura copiada.')

# 4b. Manifesto SHA-256 da Célula Extra 2
for var in ('LAST_JSON_CERT', 'LAST_MD_CERT', 'LAST_SHA256SUMS'):
  v = globals().get(var)
  if v and os.path.isfile(v):
    copiar(Path(v), f'docs/provenance/{Path(v).name}')
if not any(c.startswith('docs/provenance/') for c in copiados):
  print('  [aviso] Manifesto da Célula Extra 2 não encontrado nesta sessão: rode a Célula Extra 2 antes.')

# 4b'. Pseudo-códigos da Célula Base (Pseudo-vista), se ela rodou nesta sessão
_pv = globals().get('PSEUDO_DIR')
if _pv and os.path.isdir(_pv):
  for p in sorted(Path(_pv).glob('*')):
    if p.is_file() and p.suffix.lower() in {'.txt', '.html'}:
      copiar(p, f'docs/pseudocodigos/{p.name}')
else:
  print('  [aviso] Pseudo-códigos não encontrados nesta sessão: rode a Célula Base (Pseudo-vista) antes, se quiser enviá-los.')

# 4c. Código de cada célula executada (e-mails e senhas ocultados)
_EMAIL = re.compile(r'[\w.+-]+@(?!users\.noreply\.github\.com)[\w-]+(?:\.[\w-]+)+')
_SEGREDO = re.compile(r'''((?:pass(?:word)?|senha|pwd|token|secret|api_?key)\w*\s*=\s*)(['"]).+?\2''', re.I)


def ocultar(s: str) -> str:
  return _EMAIL.sub('[e-mail oculto]', _SEGREDO.sub(r'\1\2[OCULTO]\2', s))


if EXPORTAR_CELULAS:
  ip = get_ipython() if 'get_ipython' in globals() else None
  ultima = {}
  for src in (ip.user_ns.get('_ih', []) if ip is not None else []):
    m = re.search(r'^#\s*C[ée]lula\s+([^:\n]+)', src or '', re.M)
    if m:
      ultima[m.group(1).strip()] = src
  for rot, src in ultima.items():
    slug = re.sub(r'[^0-9A-Za-z]+', '_', rot.lower().replace('é', 'e')).strip('_')
    dst = CLONE_DIR / 'src' / 'celulas' / f'celula_{slug}.py'
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(ocultar(src).rstrip() + '\n', encoding='utf-8')
    copiados.append(dst.relative_to(CLONE_DIR).as_posix())

# 4d. O notebook (a última versão SALVA no Drive: use Ctrl+S antes de rodar esta célula)
if not (NOTEBOOK_PATH and os.path.isfile(NOTEBOOK_PATH)):
  for cand in (Path('/content/drive/MyDrive/Colab Notebooks') / NOTEBOOK_NAME, Path('/content/drive/MyDrive') / NOTEBOOK_NAME):
    if cand.is_file():
      NOTEBOOK_PATH = str(cand)
      break
if NOTEBOOK_PATH and os.path.isfile(NOTEBOOK_PATH):
  copiar(Path(NOTEBOOK_PATH), f'notebooks/{Path(NOTEBOOK_PATH).name}')
else:
  print('  [aviso] Notebook não encontrado no Drive (defina DETECTED_NB_PATH).')

# 4e. Certificados antigos que listam caminhos pessoais do Drive (versão anterior da Célula Extra 2)
removidos = []
if REMOVER_CERTIFICADOS_COM_CAMINHOS:
  for p in sorted((CLONE_DIR / 'docs' / 'provenance').glob('certificado_autoria_*')):
    try:
      if '/content/drive/' in p.read_text(encoding='utf-8', errors='ignore') or '"relative_path": "drive/' in p.read_text(encoding='utf-8', errors='ignore'):
        git('rm', '-q', str(p.relative_to(CLONE_DIR)))
        removidos.append(p.name)
    except OSError:
      pass

# 4f. README mínimo, só se o repositório ainda não tiver um (nunca sobrescreve o seu)
if CRIAR_README_SE_FALTAR and not any((CLONE_DIR / n).exists() for n in ('README.md', 'README.rst', 'README')):
  (CLONE_DIR / 'README.md').write_text(
      '# F-PluvIA Metropolitan\n\n'
      'Nowcasting de precipitação na Região Metropolitana de Fortaleza com o GPTCast-8x8 ajustado em séries '
      'GPM IMERG reamostradas a 2 km.\n\n'
      'Trabalho de Conclusão de Curso — Bacharelado em Ciência da Computação, IFCE Campus Maracanaú (2026.2).  \n'
      'Autor: Raul Santiago Pinheiro · Orientador: Amauri Holanda de Souza Júnior\n\n'
      '## Estrutura\n\n'
      '| Pasta | Conteúdo |\n|---|---|\n'
      '| `notebooks/` | Notebook do Google Colab com todas as células |\n'
      '| `src/celulas/` | Código de cada célula, um arquivo por célula |\n'
      '| `results/<ambiente>/tables/` | Tabelas de métricas (.csv) |\n'
      '| `results/<ambiente>/figures/` | Figuras e mapas (.png) |\n'
      '| `docs/provenance/` | Manifesto SHA-256 dos artefatos (conferir com `sha256sum -c`) |\n\n'
      'Os checkpoints do modelo e as séries .npz não estão no repositório (tamanho); ficam no Google Drive do autor.\n',
      encoding='utf-8')
  copiados.append('README.md')

# ----------------------------------------- 5. verificação de dados sensíveis antes do commit
print('[3/5] Verificando dados sensíveis ...')
git('add', '-A')
alterados = [l[3:] for l in git('status', '--porcelain').splitlines() if l and not l.startswith('D ')]
proibidos = {s for s in (GITHUB_TOKEN, os.environ.get('PPS_USERNAME', ''), os.environ.get('PPS_PASSWORD', '')) if len(s) >= 6}
try:
  from google.colab import userdata as _ud
  for k in ('PPS_USERNAME', 'PPS_PASSWORD'):
    try:
      v = (_ud.get(k) or '').strip()
      if len(v) >= 6:
        proibidos.add(v)
    except Exception:
      pass
except Exception:
  pass
padroes = [re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}'), re.compile(r'github_pat_[A-Za-z0-9_]{40,}')]
problemas, avisos_email = [], []
for rel in alterados:
  p = CLONE_DIR / rel.strip('"')
  if not p.is_file() or p.suffix.lower() in {'.png', '.pdf'}:
    continue
  txt = p.read_text(encoding='utf-8', errors='ignore')
  if any(s in txt for s in proibidos) or any(r.search(txt) for r in padroes):
    problemas.append(f'{rel}: contém credencial (token ou usuário/senha da NASA PPS)')
  if rel.startswith('docs/provenance/') and '/content/drive/' in txt:
    problemas.append(f'{rel}: lista caminhos do Google Drive')
  if _EMAIL.search(txt):
    avisos_email.append(rel)
if problemas:
  git('reset', '-q')
  raise RuntimeError('Envio cancelado; nada foi enviado ao GitHub:\n  - ' + '\n  - '.join(problemas)
                     + '\nNo notebook, troque o valor escrito no código por userdata.get(...) e salve antes de rodar de novo.')
if avisos_email:
  print('  [aviso] Há endereços de e-mail em: ' + ', '.join(avisos_email[:8]) + (' ...' if len(avisos_email) > 8 else ''))

# ------------------------------------------------------------------ 6. commit e push
print(f'[4/5] {len(copiados)} arquivos copiados, {len(alterados)} novos ou alterados, {len(removidos)} removidos.')
if grandes:
  print('  Acima de 95 MB (ficam só no Drive): ' + ', '.join(f'{n} ({mb} MB)' for n, mb in grandes))
if removidos:
  print('  Certificados antigos com caminhos do Drive retirados do repositório: ' + ', '.join(removidos))
  print('  (continuam no histórico do git; para apagá-los de vez é preciso reescrever o histórico.)')
if not git('status', '--porcelain'):
  print('[5/5] Nada mudou: o repositório já está sincronizado.')
elif SO_PREPARAR:
  print('[5/5] SO_PREPARAR = True: commit não criado. Arquivos que seriam enviados:')
  print('\n'.join('  ' + l for l in git('status', '--porcelain').splitlines()[:60]))
  git('reset', '-q')
else:
  agora = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3)))
  raiz = ''
  if globals().get('LAST_JSON_CERT') and os.path.isfile(LAST_JSON_CERT):
    try:
      raiz = json.loads(Path(LAST_JSON_CERT).read_text(encoding='utf-8'))['artefatos']['hash_raiz'][:16]
    except (KeyError, json.JSONDecodeError):
      pass
  corpo = (f'Sincronização automática do Google Colab ({agora:%Y-%m-%d %H:%M} BRT).\n\n'
           f'- Ambiente: {ENV_TAG}\n- Arquivos novos ou alterados: {len(alterados)}\n'
           + (f'- Hash raiz do manifesto (início): {raiz}\n' if raiz else '')
           + (f'- Certificados antigos removidos: {len(removidos)}\n' if removidos else ''))
  git('commit', '-q', '-m', f'chore(results): sincroniza artefatos de {ENV_TAG}', '-m', corpo)
  print(f'[5/5] Enviando para o ramo {RAMO} ...')
  if not REPO_VAZIO:
    git('pull', '-q', '--rebase', 'origin', RAMO, auth=True)
  git('push', '-q', '-u', 'origin', f'HEAD:{RAMO}', auth=True)
  print(f'[Deploy] Concluído: {REPO_URL.replace(".git", "")}/commit/{git("rev-parse", "HEAD")}')
shutil.rmtree(CLONE_DIR, ignore_errors=True)
