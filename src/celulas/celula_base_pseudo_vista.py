# @title
# Célula Base (Pseudo-vista): traduz cada célula de código do notebook para pseudo-código em português
# (tradução por regras, a partir da árvore sintática do Python), grava um .txt por célula na pasta
# "Pseudo-códigos" do Drive e gera um HTML que copia o notebook com as células de código trocadas pelo
# pseudo-código.
#
import ast
import copy
import datetime
import glob
import io
import json
import os
import re
import tokenize
import unicodedata
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter

ENV_TAG = globals().get('ENV_TAG', 'Ambiente_Ativo')
SERIES_CACHE_DIR = globals().get('SERIES_CACHE_DIR', '/content/drive/MyDrive/GPTCast_RMF/series_2019_2025_2km')
NOTEBOOK_NAME = 'Model_Annual_Full_Training_Series_TCC2_-_2026_2_-_RAUL_SANTIAGO_PINHEIRO.ipynb'
NOTEBOOK_PATH = globals().get('DETECTED_NB_PATH')
PSEUDO_DIR = Path(os.path.dirname(SERIES_CACHE_DIR.rstrip('/'))) / 'Pseudo-códigos'
MAX_LINHA = 150            # linhas de pseudo-código mais longas são cortadas com "…"
MAX_TEXTO = 60             # textos literais mais longos aparecem resumidos

# ------------------------------------------------------------------ 1. o notebook salvo
if not (NOTEBOOK_PATH and os.path.isfile(NOTEBOOK_PATH)):
  for cand in (Path('/content/drive/MyDrive/Colab Notebooks') / NOTEBOOK_NAME, Path('/content/drive/MyDrive') / NOTEBOOK_NAME):
    if cand.is_file():
      NOTEBOOK_PATH = str(cand)
      break
if not (NOTEBOOK_PATH and os.path.isfile(NOTEBOOK_PATH)):
  raise RuntimeError(f'Notebook {NOTEBOOK_NAME} não encontrado no Drive. Salve (Ctrl+S) ou defina DETECTED_NB_PATH.')
nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
salvo = datetime.datetime.fromtimestamp(os.path.getmtime(NOTEBOOK_PATH))
print(f'[Pseudo-vista] Notebook: {os.path.basename(NOTEBOOK_PATH)} (salvo em {salvo:%d/%m/%Y %H:%M}; '
      'use Ctrl+S antes, se tiver mudado algo).')

# ------------------------------------------------- 2. privacidade (antes de traduzir)
_EMAIL = re.compile(r'[\w.+-]+@(?!users\.noreply\.github\.com)[\w-]+(?:\.[\w-]+)+')
_SEGREDO = re.compile(r'''((?:pass(?:word)?|senha|pwd|token|secret|api_?key)\w*\s*=\s*)(['"]).+?\2''', re.I)


def ocultar(s: str) -> str:
  return _EMAIL.sub('[e-mail oculto]', _SEGREDO.sub(r'\1\2[OCULTO]\2', s))


# ------------------------------------------------------ 3. tradutor Python -> pseudo-código
OPS = {ast.Add: '+', ast.Sub: '−', ast.Mult: '×', ast.Div: '/', ast.FloorDiv: 'DIV', ast.Mod: 'MOD',
       ast.Pow: '^', ast.MatMult: '@', ast.BitAnd: '&', ast.BitOr: '|', ast.BitXor: 'XOR',
       ast.LShift: '<<', ast.RShift: '>>'}
CMP = {ast.Eq: '=', ast.NotEq: '≠', ast.Lt: '<', ast.LtE: '≤', ast.Gt: '>', ast.GtE: '≥',
       ast.In: 'EM', ast.NotIn: 'NÃO EM', ast.Is: 'É', ast.IsNot: 'NÃO É'}
EXIBIR = {'print', 'display', 'plt.show'}


def E(n) -> str:
  """Expressão em pseudo-código."""
  if n is None:
    return ''
  t = type(n)
  if t is ast.Constant:
    v = n.value
    if v is None: return 'NULO'
    if v is True: return 'VERDADEIRO'
    if v is False: return 'FALSO'
    if isinstance(v, str):
      v = v.replace('\n', ' ')
      return f'"{v[:MAX_TEXTO]}…"' if len(v) > MAX_TEXTO else f'"{v}"'
    return repr(v)
  if t is ast.Name: return n.id
  if t is ast.Attribute: return f'{E(n.value)}.{n.attr}'
  if t is ast.Subscript: return f'{E(n.value)}[{E(n.slice)}]'
  if t is ast.Slice: return f'{E(n.lower)}:{E(n.upper)}' + (f':{E(n.step)}' if n.step else '')
  if t is ast.Starred: return f'*{E(n.value)}'
  if t is ast.Call:
    a = [E(x) for x in n.args] + [(f'{k.arg}={E(k.value)}' if k.arg else f'**{E(k.value)}') for k in n.keywords]
    return f'{E(n.func)}({", ".join(a)})'
  if t is ast.BinOp: return f'{E(n.left)} {OPS.get(type(n.op), "?")} {E(n.right)}'
  if t is ast.UnaryOp:
    return {ast.Not: f'NÃO {E(n.operand)}', ast.USub: f'−{E(n.operand)}', ast.UAdd: E(n.operand),
            ast.Invert: f'~{E(n.operand)}'}[type(n.op)]
  if t is ast.BoolOp: return f' {"E" if isinstance(n.op, ast.And) else "OU"} '.join(f'({E(v)})' if isinstance(v, ast.BoolOp) else E(v) for v in n.values)
  if t is ast.Compare:
    s = E(n.left)
    for op, c in zip(n.ops, n.comparators):
      s += ' É NULO' if isinstance(op, ast.Is) and isinstance(c, ast.Constant) and c.value is None else \
           ' NÃO É NULO' if isinstance(op, ast.IsNot) and isinstance(c, ast.Constant) and c.value is None else \
           f' {CMP[type(op)]} {E(c)}'
    return s
  if t is ast.IfExp: return f'({E(n.body)} SE {E(n.test)} SENÃO {E(n.orelse)})'
  if t in (ast.List, ast.Tuple, ast.Set):
    a, b = {ast.List: '[]', ast.Tuple: '()', ast.Set: '{}'}[t]
    return a + ', '.join(E(x) for x in n.elts) + b
  if t is ast.Dict:
    return '{' + ', '.join((f'{E(k)}: {E(v)}' if k is not None else f'**{E(v)}') for k, v in zip(n.keys, n.values)) + '}'
  if t in (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp):
    elt = f'{E(n.key)}: {E(n.value)}' if t is ast.DictComp else E(n.elt)
    gens = ' '.join(f'PARA CADA {E(g.target)} EM {E(g.iter)}' + ''.join(f' SE {E(c)}' for c in g.ifs) for g in n.generators)
    return f'[{elt} {gens}]'
  if t is ast.Lambda: return f'FUNÇÃO({", ".join(a.arg for a in n.args.args)}) ↦ {E(n.body)}'
  if t is ast.NamedExpr: return f'({E(n.target)} ← {E(n.value)})'
  if t in (ast.Await,): return f'AGUARDAR {E(n.value)}'
  if t is ast.JoinedStr:
    s = ast.unparse(n).replace('\n', ' ')
    return s[:MAX_TEXTO + 4] + '…' if len(s) > MAX_TEXTO + 4 else s
  try:
    return ast.unparse(n)
  except Exception:
    return '…'


def _args(a: ast.arguments) -> str:
  nomes = [x.arg for x in a.posonlyargs + a.args]
  if a.vararg: nomes.append('*' + a.vararg.arg)
  nomes += [x.arg for x in a.kwonlyargs]
  if a.kwarg: nomes.append('**' + a.kwarg.arg)
  return ', '.join(nomes)


class Tradutor:
  def __init__(self, src: str):
    self.out, self.comentarios, self.usados = [], {}, set()
    try:
      for tk in tokenize.generate_tokens(io.StringIO(src).readline):
        if tk.type == tokenize.COMMENT:
          txt = tk.string.lstrip('#').strip()
          if txt and not txt.startswith('@'):
            self.comentarios[tk.start[0]] = (txt, tk.line.strip().startswith('#'))
    except (tokenize.TokenError, IndentationError, SyntaxError):
      pass

  def w(self, nivel, txt):
    linha = '    ' * nivel + txt
    self.out.append(linha if len(linha) <= MAX_LINHA else linha[:MAX_LINHA - 1] + '…')

  def coment_antes(self, nivel, ate):
    for ln in sorted(k for k in self.comentarios if k < ate and k not in self.usados):
      txt, cheio = self.comentarios[ln]
      if cheio:
        self.w(nivel, f'// {txt}')
      self.usados.add(ln)

  def inline(self, ln):
    c = self.comentarios.get(ln)
    if c and not c[1] and ln not in self.usados:
      self.usados.add(ln)
      return f'   // {c[0]}'
    return ''

  def bloco(self, corpo, nivel):
    for st in corpo:
      self.st(st, nivel)

  def st(self, n, nv):
    self.coment_antes(nv, n.lineno)
    tail = self.inline(n.lineno)
    t = type(n)
    if t in (ast.Import, ast.ImportFrom):
      mod = (('.' * n.level) + (n.module or '')) if t is ast.ImportFrom else ''
      nomes = ', '.join(a.name + (f' (como {a.asname})' if a.asname else '') for a in n.names)
      self.w(nv, (f'IMPORTAR {nomes} DE {mod}' if mod else f'IMPORTAR {nomes}') + tail)
    elif t in (ast.FunctionDef, ast.AsyncFunctionDef):
      for d in n.decorator_list:
        self.w(nv, f'// decorador: {E(d)}')
      self.w(nv, f'FUNÇÃO {n.name}({_args(n.args)})' + tail)
      corpo = n.body
      doc = ast.get_docstring(n)
      if doc:
        for ln in doc.strip().splitlines():
          self.w(nv + 1, f'// {ln.strip()}')
        corpo = corpo[1:]
      self.bloco(corpo or [ast.Pass(lineno=n.lineno)], nv + 1)
      self.w(nv, 'FIM FUNÇÃO')
    elif t is ast.ClassDef:
      base = f' (herda de {", ".join(E(b) for b in n.bases)})' if n.bases else ''
      self.w(nv, f'CLASSE {n.name}{base}' + tail)
      corpo = n.body
      doc = ast.get_docstring(n)
      if doc:
        for ln in doc.strip().splitlines():
          self.w(nv + 1, f'// {ln.strip()}')
        corpo = corpo[1:]
      self.bloco(corpo, nv + 1)
      self.w(nv, 'FIM CLASSE')
    elif t is ast.If:
      self.w(nv, f'SE {E(n.test)} ENTÃO' + tail)
      self.bloco(n.body, nv + 1)
      orelse = n.orelse
      while len(orelse) == 1 and isinstance(orelse[0], ast.If):
        self.coment_antes(nv, orelse[0].lineno)
        self.w(nv, f'SENÃO SE {E(orelse[0].test)} ENTÃO')
        self.bloco(orelse[0].body, nv + 1)
        orelse = orelse[0].orelse
      if orelse:
        self.w(nv, 'SENÃO')
        self.bloco(orelse, nv + 1)
      self.w(nv, 'FIM SE')
    elif t in (ast.For, ast.AsyncFor):
      self.w(nv, f'PARA CADA {E(n.target)} EM {E(n.iter)} FAÇA' + tail)
      self.bloco(n.body, nv + 1)
      if n.orelse:
        self.w(nv, 'SE O LAÇO TERMINAR SEM INTERRUPÇÃO')
        self.bloco(n.orelse, nv + 1)
      self.w(nv, 'FIM PARA')
    elif t is ast.While:
      self.w(nv, f'ENQUANTO {E(n.test)} FAÇA' + tail)
      self.bloco(n.body, nv + 1)
      self.w(nv, 'FIM ENQUANTO')
    elif t in (ast.With, ast.AsyncWith):
      itens = ', '.join(E(i.context_expr) + (f' COMO {E(i.optional_vars)}' if i.optional_vars else '') for i in n.items)
      self.w(nv, f'USANDO {itens}' + tail)
      self.bloco(n.body, nv + 1)
      self.w(nv, 'FIM USANDO')
    elif t is ast.Try or t.__name__ == 'TryStar':
      self.w(nv, 'TENTAR' + tail)
      self.bloco(n.body, nv + 1)
      for h in n.handlers:
        self.w(nv, f'SE OCORRER ERRO{" " + E(h.type) if h.type else ""}{" (chamado " + h.name + ")" if h.name else ""}')
        self.bloco(h.body, nv + 1)
      if n.orelse:
        self.w(nv, 'SE NÃO OCORRER ERRO')
        self.bloco(n.orelse, nv + 1)
      if n.finalbody:
        self.w(nv, 'AO FINAL, SEMPRE')
        self.bloco(n.finalbody, nv + 1)
      self.w(nv, 'FIM TENTAR')
    elif t is ast.Return:
      self.w(nv, f'RETORNAR {E(n.value)}'.rstrip() + tail)
    elif t is ast.Raise:
      self.w(nv, f'LANÇAR ERRO {E(n.exc)}'.rstrip() + tail)
    elif t is ast.Assign:
      alvos = ' ← '.join(E(a) for a in n.targets)
      self.w(nv, f'{alvos} ← {E(n.value)}' + tail)
    elif t is ast.AugAssign:
      self.w(nv, f'{E(n.target)} ← {E(n.target)} {OPS.get(type(n.op), "?")} {E(n.value)}' + tail)
    elif t is ast.AnnAssign:
      self.w(nv, (f'{E(n.target)} ← {E(n.value)}' if n.value is not None else f'DECLARAR {E(n.target)}') + tail)
    elif t is ast.Expr:
      v = n.value
      if isinstance(v, ast.Constant) and isinstance(v.value, str):
        for ln in v.value.strip().splitlines():
          self.w(nv, f'// {ln.strip()}')
      elif isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == '__SHELL__':
        self.w(nv, f'EXECUTAR NO TERMINAL: {v.args[0].value}' + tail)
      elif isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == '__MAGIC__':
        self.w(nv, f'COMANDO DO AMBIENTE: {v.args[0].value}' + tail)
      elif isinstance(v, ast.Call) and E(v.func) in EXIBIR:
        self.w(nv, f'EXIBIR {", ".join(E(a) for a in v.args)}'.rstrip() + tail)
      else:
        self.w(nv, f'EXECUTAR {E(v)}' + tail)
    elif t is ast.Pass:
      self.w(nv, '(nada a fazer)')
    elif t is ast.Break:
      self.w(nv, 'INTERROMPER LAÇO')
    elif t is ast.Continue:
      self.w(nv, 'PASSAR À PRÓXIMA ITERAÇÃO')
    elif t is ast.Assert:
      self.w(nv, f'GARANTIR QUE {E(n.test)}' + (f' (senão: {E(n.msg)})' if n.msg else ''))
    elif t is ast.Delete:
      self.w(nv, f'REMOVER {", ".join(E(x) for x in n.targets)}' + tail)
    elif t in (ast.Global, ast.Nonlocal):
      self.w(nv, f'USAR VARIÁVEIS {"GLOBAIS" if t is ast.Global else "EXTERNAS"} {", ".join(n.names)}')
    else:
      try:
        self.w(nv, ast.unparse(n).splitlines()[0] + ' …')
      except Exception:
        self.w(nv, '…')


def _preparar(src: str) -> str:
  """Troca linhas de terminal (!) e comandos mágicos (%) por chamadas marcadoras, mantendo a indentação."""
  linhas = []
  for ln in src.splitlines():
    s = ln.lstrip()
    ind = ln[:len(ln) - len(s)]
    if s.startswith('!'):
      linhas.append(f'{ind}__SHELL__({s[1:].strip()!r})')
    elif s.startswith('%'):
      linhas.append(f'{ind}__MAGIC__({s.lstrip("%").strip()!r})')
    else:
      linhas.append(ln)
  return '\n'.join(linhas)


def traduzir(src: str) -> str:
  src = _preparar(ocultar(src))
  try:
    arvore = ast.parse(src)
  except SyntaxError as e:
    return f'// Pseudo-código indisponível: a célula não é Python válido (linha {e.lineno}: {e.msg}).'
  tr = Tradutor(src)
  for st in arvore.body:
    tr.st(st, 0)
  tr.coment_antes(0, 10 ** 9)
  return '\n'.join(tr.out) if tr.out else '// (célula sem instruções)'


# ---------------------------------------------------------------- 4. traduz as células
_ROT = re.compile(r'^#\s*(C[ée]lula\s+[^:\n]+)', re.M)
PSEUDO_DIR.mkdir(parents=True, exist_ok=True)
for antigo in glob.glob(str(PSEUDO_DIR / '*.txt')) + glob.glob(str(PSEUDO_DIR / '*.html')):
  os.remove(antigo)                          # a pasta reflete sempre o notebook atual

nb_pseudo = copy.deepcopy(nb)
indice, n_cod = [], 0
for i, cel in enumerate(nb_pseudo.cells):
  if cel.cell_type != 'code' or not cel.source.strip():
    continue
  n_cod += 1
  m = _ROT.search(cel.source)
  rotulo = m.group(1).strip() if m else f'Célula de código nº {n_cod}'
  pseudo = traduzir(cel.source)
  cel.source = f'// ===== {rotulo} — pseudo-código =====\n{pseudo}'
  slug = re.sub(r'[^0-9a-z]+', '_', unicodedata.normalize('NFKD', rotulo.lower()).encode('ascii', 'ignore').decode()).strip('_')
  nome = f'{n_cod:02d}_{slug}.txt'
  (PSEUDO_DIR / nome).write_text(cel.source + '\n', encoding='utf-8')
  indice.append({'Ordem': n_cod, 'Célula': rotulo, 'Arquivo': nome,
                 'Linhas de código': len(nb.cells[i].source.splitlines()), 'Linhas de pseudo-código': len(pseudo.splitlines())})

# saídas de texto: mesmas regras de privacidade
for cel in nb_pseudo.cells:
  for out in cel.get('outputs', []):
    if 'text' in out:
      out['text'] = ocultar(out['text'] if isinstance(out['text'], str) else ''.join(out['text']))
    for k in ('text/plain', 'text/html', 'text/markdown'):
      if k in out.get('data', {}):
        v = out['data'][k]
        out['data'][k] = ocultar(v if isinstance(v, str) else ''.join(v))

LEGENDA = ('// Convenções: ← atribuição | FUNÇÃO/CLASSE/SE/PARA CADA/ENQUANTO/TENTAR … FIM | EXIBIR = mostra na saída\n'
           '// EXECUTAR = chama uma função | // = comentário do código original | e-mails e senhas ocultados.')
(PSEUDO_DIR / '00_LEIA-ME.txt').write_text(
    'Pseudo-códigos do notebook F-PluvIA Metropolitan\n'
    f'Notebook: {os.path.basename(NOTEBOOK_PATH)} (salvo em {salvo:%d/%m/%Y %H:%M})\n'
    f'Gerado em: {datetime.datetime.now():%d/%m/%Y %H:%M} | Ambiente: {ENV_TAG}\n'
    'Tradução automática por regras (árvore sintática do Python): é fiel à estrutura do código, não um resumo.\n\n'
    + LEGENDA + '\n\n' + '\n'.join(f"{r['Ordem']:02d}. {r['Célula']} -> {r['Arquivo']}" for r in indice) + '\n',
    encoding='utf-8')

# --------------------------------------------------------------------- 5. HTML
nb_pseudo.metadata['language_info'] = {'name': 'text', 'pygments_lexer': 'text'}
nb_pseudo.metadata.pop('widgets', None)
nb_pseudo.cells.insert(0, nbformat.v4.new_markdown_cell(
    '# Pseudo-vista do notebook F-PluvIA Metropolitan\n\n'
    'Cópia do notebook com o conteúdo de cada célula de código trocado pela sua tradução para pseudo-código '
    '(tradução automática por regras). Células de texto e saídas são as do notebook salvo.\n\n'
    '```\n' + LEGENDA + '\n```'))
exp = HTMLExporter()
exp.exclude_input_prompt = True
html, _ = exp.from_notebook_node(nb_pseudo)
html_path = PSEUDO_DIR / f'{Path(NOTEBOOK_PATH).stem}_pseudo_vista.html'
html_path.write_text(html, encoding='utf-8')

import pandas as pd
display(pd.DataFrame(indice).set_index('Ordem'))
print(f'[Pseudo-vista] {len(indice)} células traduzidas -> {PSEUDO_DIR}')
print(f'[Pseudo-vista] HTML -> {html_path} ({html_path.stat().st_size / 2 ** 20:.1f} MB)')
print('[Pseudo-vista] A Célula de Deploy envia esta pasta para docs/pseudocodigos/ no GitHub.')
