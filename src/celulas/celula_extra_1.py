# @title
# Célula Extra 1: Rastreabilidade do Código Original do GPTCast (DSIP-FBK/GPTCast, Franch et al., 2024)
# Destaca, com texto vermelho (#CC0000) sobre fundo amarelo (#FFEB3B), cada nome que vem da biblioteca
# gptcast instalada, mantendo a paleta de sintaxe do editor do Colab. Os nomes são lidos do próprio
# pacote instalado (análise do código-fonte, sem importar nada), não de uma lista escrita à mão.
import ast
import builtins
import html
import importlib.util
import io
import os
import re
import time
import tokenize
import token as tok_mod
from pathlib import Path

import pandas as pd
from IPython.display import HTML, display

t_ini = time.perf_counter()
ENV_TAG = globals().get('ENV_TAG', 'Ambiente_Ativo')
RESULTS_ENV_DIR = globals().get('RESULTS_ENV_DIR', os.path.join(globals().get('SERIES_CACHE_DIR', '/content'),
                                                               f'resultados_{ENV_TAG}'))
NOTEBOOK_PATH = None        # opcional: caminho de um .ipynb; se None, usa as células executadas nesta sessão
MOSTRAR_CODIGO = True       # False = só a tabela-resumo
COR_TEXTO, COR_FUNDO = '#CC0000', '#FFEB3B'


# ------------------------------------------------- 1. nomes do pacote gptcast instalado
def nomes_do_pacote(pacote: str = 'gptcast') -> dict:
  """Lê os .py do pacote e devolve módulos, classes/funções de topo e métodos/atributos das classes."""
  spec = importlib.util.find_spec(pacote)
  if spec is None or not spec.submodule_search_locations:
    raise RuntimeError('Pacote gptcast não encontrado. Rode a Célula 1 (instalação) antes.')
  raiz = Path(list(spec.submodule_search_locations)[0])
  modulos, topo, membros = {pacote}, set(), set()
  for py in raiz.rglob('*.py'):
    partes = py.relative_to(raiz.parent).with_suffix('').parts
    modulos.update(p for p in partes if p != '__init__')
    try:
      arvore = ast.parse(py.read_text(encoding='utf-8'))
    except (SyntaxError, UnicodeDecodeError):
      continue
    for no in arvore.body:
      if isinstance(no, (ast.FunctionDef, ast.ClassDef)) and not no.name.startswith('_'):
        topo.add(no.name)
      if isinstance(no, ast.ClassDef):
        for sub in ast.walk(no):
          if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and not sub.name.startswith('_'):
            membros.add(sub.name)
          # atributos de instância só das classes de modelo (GPTCast, VAEGAN...), não de dados/utilitários
          if (py.parent.name == 'models' and isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name)
              and sub.value.id == 'self' and isinstance(sub.ctx, ast.Store) and not sub.attr.startswith('_')):
            membros.add(sub.attr)
  # Métodos com nomes genéricos (str.encode, bytes.decode, forward do PyTorch, ganchos do Lightning)
  # não identificam o GPTCast quando aparecem depois de um ponto: ficam de fora.
  genericos = set(dir(str)) | set(dir(bytes)) | set(dir(dict)) | set(dir(list)) | set(dir(object))
  for mod, attr in (('torch.nn', 'Module'), ('torch', 'Tensor'), ('numpy', 'ndarray'), ('pandas', 'DataFrame'),
                    ('pandas', 'Series'), ('xarray', 'Dataset'), ('xarray', 'DataArray')):
    try:
      genericos |= set(dir(getattr(importlib.import_module(mod), attr)))
    except Exception:
      continue
  for mod in ('lightning.pytorch', 'pytorch_lightning'):
    try:
      genericos |= set(dir(importlib.import_module(mod).LightningModule))
      break
    except Exception:
      continue
  genericos |= {'forward', 'training_step', 'validation_step', 'test_step', 'configure_optimizers',
                'shared_step', 'get_input', 'log_images', 'hparams', 'config', 'model', 'models', 'data',
                'loss', 'name', 'params', 'state', 'utils', 'train', 'main', 'l', 'crop', 'seq_len', 'device', 'training'}
  return {'modulos': modulos, 'topo': topo, 'membros': membros - genericos, 'raiz': str(raiz)}


GPTCAST = nomes_do_pacote()


# ---------------------------------------------------------- 2. células a analisar
_ROTULO = re.compile(r'^#\s*(C[ée]lula\s+[^:\n]+)', re.M)


def _rotulo(src: str, padrao: str) -> str:
  m = _ROTULO.search(src)
  return m.group(1).strip() if m else padrao


def celulas() -> list:
  """Células desta sessão (última execução de cada uma, pelo rótulo '# Célula X:') ou de um .ipynb."""
  if NOTEBOOK_PATH:
    import json
    with open(NOTEBOOK_PATH, encoding='utf-8') as f:
      nb = json.load(f)
    out = []
    for i, c in enumerate(nb.get('cells', [])):
      if c.get('cell_type') == 'code':
        src = ''.join(c.get('source', [])) if isinstance(c.get('source'), list) else str(c.get('source', ''))
        if src.strip():
          out.append((_rotulo(src, f'Célula de código nº {i + 1}'), src))
    return out
  ip = get_ipython()
  hist = ip.user_ns.get('_ih', []) if ip is not None else []
  ultima = {}
  for i, src in enumerate(hist):
    if isinstance(src, str) and src.strip():
      ultima[_rotulo(src, f'Execução In[{i}] sem rótulo')] = (i, src)
  return [(k, s) for k, (i, s) in sorted(ultima.items(), key=lambda kv: kv[1][0])]


# ------------------------------------------- 3. privacidade: nada sensível vai para a saída
_EMAIL = re.compile(r'[\w.+-]+@[\w-]+(?:\.[\w-]+)+')
_SEGREDO = re.compile(r'''((?:pass(?:word)?|senha|pwd|token|secret|api_?key)\w*\s*=\s*)(['"])(.+?)\2''', re.I)


def ocultar(src: str) -> str:
  src = _SEGREDO.sub(lambda m: f'{m.group(1)}{m.group(2)}[OCULTO]{m.group(2)}', src)
  return _EMAIL.sub('[e-mail oculto]', src)


# -------------------------------------------------------------- 4. destaque sintático
KW_FLUXO = {'import', 'from', 'as', 'return', 'if', 'elif', 'else', 'for', 'while', 'try', 'except', 'finally',
            'with', 'raise', 'pass', 'break', 'continue', 'yield'}
KW_DECL = {'def', 'class', 'lambda', 'None', 'True', 'False', 'and', 'or', 'not', 'in', 'is', 'global',
           'nonlocal', 'assert', 'del', 'async', 'await', 'self', 'cls'}
BUILTINS = set(dir(builtins))


def importados_do_gptcast(src: str) -> set:
  """Nomes que a célula traz do gptcast (inclusive apelidos de 'as')."""
  nomes = set()
  try:
    arvore = ast.parse(src)
  except SyntaxError:
    arvore = None
  if arvore is not None:
    for no in ast.walk(arvore):
      if isinstance(no, ast.ImportFrom) and (no.module or '').split('.')[0] == 'gptcast':
        nomes.update(a.asname or a.name for a in no.names)
      elif isinstance(no, ast.Import):
        nomes.update((a.asname or a.name).split('.')[0] for a in no.names if a.name.split('.')[0] == 'gptcast')
  return nomes


def destacar(src: str, nomes_sessao: set) -> tuple:
  """Devolve (html, ocorrências, nomes encontrados, nomes do gptcast reatribuídos no notebook)."""
  linhas = src.splitlines(keepends=True)
  offs = [0]
  for ln in linhas:
    offs.append(offs[-1] + len(ln))
  partes, achados, alterados = [], [], set()
  pos, ant, ant2, linha_import, raiz_import = 0, '', '', False, ''
  toks = []
  try:
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
  except (tokenize.TokenError, IndentationError, SyntaxError):
    pass                                           # células com %magic ou !shell: o resto fica sem cor
  for j, (tt, ts, (sr, sc), (er, ec), _) in enumerate(toks):
    if tt in (tok_mod.ENDMARKER, tok_mod.INDENT, tok_mod.DEDENT) or sr - 1 >= len(offs) - 1:
      continue
    if tt in (tok_mod.NEWLINE, tok_mod.NL):
      linha_import = False
    a, b = offs[sr - 1] + sc, offs[er - 1] + ec
    if a > pos:
      partes.append(html.escape(src[pos:a]))
    e = html.escape(ts)
    if tt == tok_mod.NAME:
      if linha_import and ant in ('from', 'import'):
        raiz_import = ts
      upstream = ((linha_import and raiz_import == 'gptcast' and (ts in GPTCAST['modulos'] or ts in GPTCAST['topo']))
                  or (ant not in ('.', 'def', 'class') and ts in nomes_sessao)
                  or (ant == '.' and (ts in GPTCAST['membros'] or (ant2 in nomes_sessao and ts in GPTCAST['topo']))))
      if ts in ('import', 'from') and ant == '':
        linha_import, raiz_import = True, ''
      if upstream:
        prox = toks[j + 1][1] if j + 1 < len(toks) else ''
        if ant == '.' and ant2 in nomes_sessao and prox == '=':
          alterados.add(f'{ant2}.{ts}')
          partes.append(f'<span class="gc-up gc-mod" title="nome do GPTCast reatribuído neste notebook">{e}</span>')
        else:
          partes.append(f'<span class="gc-up">{e}</span>')
        achados.append(ts)
      elif ts in KW_FLUXO:
        partes.append(f'<span class="k-fl">{e}</span>')
      elif ts in KW_DECL:
        partes.append(f'<span class="k-dc">{e}</span>')
      elif ant == 'class':
        partes.append(f'<span class="k-cl">{e}</span>')
      elif ant == 'def' or ts in BUILTINS:
        partes.append(f'<span class="k-fn">{e}</span>')
      else:
        partes.append(e)
    elif tt == tok_mod.COMMENT:
      partes.append(f'<span class="k-cm">{e}</span>')
    elif tt == tok_mod.STRING:
      partes.append(f'<span class="k-st">{e}</span>')
    elif tt == tok_mod.NUMBER:
      partes.append(f'<span class="k-nu">{e}</span>')
    else:
      partes.append(e)
    pos = b
    if tt not in (tok_mod.NL, tok_mod.NEWLINE, tok_mod.COMMENT):
      ant2, ant = ant, ts
    if tt in (tok_mod.NEWLINE, tok_mod.NL):
      ant2, ant = '', ''
  if pos < len(src):
    partes.append(html.escape(src[pos:]))
  return ''.join(partes), len(achados), sorted(set(achados)), sorted(alterados)


# --------------------------------------------------------------------- 5. execução
cels = [(r, ocultar(s)) for r, s in celulas() if not r.lower().startswith(('célula extra 1', 'celula extra 1'))]
nomes_sessao = {'gptcast'}
for _, s in cels:
  nomes_sessao |= importados_do_gptcast(s)

linhas_tab, blocos, total = [], [], 0
for rot, src in cels:
  h, n, uniq, alt = destacar(src, nomes_sessao)
  if n == 0:
    continue
  total += n
  linhas_tab.append({'Célula': rot, 'Ocorrências': n, 'Nomes do GPTCast usados': ', '.join(uniq),
                     'Reatribuídos no notebook': ', '.join(alt)})
  if MOSTRAR_CODIGO:
    badges = ' '.join(f'<span class="gc-badge">{html.escape(t)}</span>' for t in uniq)
    aviso = (f'<div class="gc-bar">Reatribuídos aqui (o notebook altera o original): '
             f'{", ".join(html.escape(a) for a in alt)}</div>') if alt else ''
    blocos.append(f'<details class="gc-box"><summary class="gc-head"><span>📄 <b>{html.escape(rot)}</b></span>'
                  f'<span class="gc-count">{n} ocorrências</span></summary>'
                  f'<div class="gc-bar">Nomes do GPTCast: {badges}</div>{aviso}'
                  f'<pre class="gc-code"><code>{h}</code></pre></details>')

df_rastro = pd.DataFrame(linhas_tab)
os.makedirs(RESULTS_ENV_DIR, exist_ok=True)
csv = os.path.join(RESULTS_ENV_DIR, f'rastreabilidade_gptcast_por_celula_{ENV_TAG}.csv')
df_rastro.to_csv(csv, index=False)
ms = (time.perf_counter() - t_ini) * 1000

CSS = f"""<style>
.gc-wrap{{font-family:'JetBrains Mono','Fira Code',Consolas,monospace;background:#1e1e1e;color:#d4d4d4;
  border:1px solid #333;border-radius:8px;padding:14px;margin-top:6px}}
.gc-title{{font-size:13.5px;font-weight:bold;color:#fff;border-bottom:1px solid #333;padding-bottom:9px;
  margin-bottom:10px;display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}}
.k-cm{{color:#6a9955;font-style:italic}} .k-fl{{color:#c586c0;font-weight:600}} .k-dc{{color:#569cd6;font-weight:600}}
.k-fn{{color:#dcdcaa}} .k-cl{{color:#4ec9b0}} .k-st{{color:#ce9178}} .k-nu{{color:#b5cea8}}
.gc-up{{color:{COR_TEXTO}!important;background:{COR_FUNDO}!important;font-weight:800;padding:0 3px;border-radius:2px}}
.gc-mod{{outline:2px dashed {COR_TEXTO};outline-offset:1px}}
.gc-badge,.gc-count{{display:inline-block;background:{COR_FUNDO};color:{COR_TEXTO};border:1px solid {COR_TEXTO};
  border-radius:4px;padding:1px 6px;margin:2px;font-size:11px;font-weight:800}}
.gc-box{{background:#252526;border:1px solid #3c3c3c;border-radius:6px;margin-bottom:9px;overflow:hidden}}
.gc-head{{cursor:pointer;padding:7px 12px;background:#2d2d30;font-size:12.5px;display:flex;justify-content:space-between}}
.gc-bar{{padding:5px 12px;background:#1e1e1e;border-bottom:1px solid #333;font-size:11.5px;color:#9cdcfe}}
.gc-code{{margin:0;padding:12px;font-size:12px;line-height:1.55;overflow-x:auto;background:#1e1e1e;color:#d4d4d4}}
</style>"""
fonte = 'arquivo ' + os.path.basename(NOTEBOOK_PATH) if NOTEBOOK_PATH else 'células executadas nesta sessão'
corpo = ''.join(blocos) if blocos else ('<p>Nenhuma célula com nomes do GPTCast. Execute as células do notebook '
                                        'ou defina NOTEBOOK_PATH.</p>' if not linhas_tab else '')
display(HTML(f'{CSS}<div class="gc-wrap"><div class="gc-title"><span>🔍 Código original '
             f'<span class="gc-badge">DSIP-FBK/GPTCast</span> no notebook ({fonte})</span>'
             f'<span class="gc-count">{total} ocorrências em {len(linhas_tab)} células · {ms:.0f} ms</span></div>'
             f'{corpo}</div>'))
print(f'[Célula Extra 1] Nomes lidos do pacote em {GPTCAST["raiz"]}: {len(GPTCAST["modulos"])} módulos, '
      f'{len(GPTCAST["topo"])} classes/funções, {len(GPTCAST["membros"])} métodos/atributos específicos.')
print('[Célula Extra 1] E-mails e valores atribuídos a senha/token/chave aparecem como [OCULTO] na saída.')
display(df_rastro.set_index('Célula') if len(df_rastro) else df_rastro)
print(f'[Célula Extra 1] Tabela -> {csv}')
