# @title
# Célula Full UMLs: Diagramas UML Gerados do Código de Cada Célula — classes, operações, variáveis produzidas,
# bibliotecas usadas e o fluxo de dados entre as células, lidos do notebook salvo (CTRL+S :necessita salvar o ambiente da execusão dessa célula).
import ast
import builtins
import html
import io
import json
import os
import re
import shutil
import subprocess
import tokenize
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# ------------------------------------------------------------------- 1. configuração
NOTEBOOK_NAME = 'Model_Annual_Full_Training_Series_TCC2_-_2026_2_-_RAUL_SANTIAGO_PINHEIRO.ipynb'
NOTEBOOK_PATH = globals().get('DETECTED_NB_PATH')          # se None, procura NOTEBOOK_NAME no Drive
SERIES_CACHE_DIR = globals().get('SERIES_CACHE_DIR', '/content/drive/MyDrive/GPTCast_RMF/series_2019_2025_2km')
UML_DIR = Path(SERIES_CACHE_DIR) / 'diagramas_uml'        # o Deploy publica esta pasta em docs/uml/
FORMATOS = ('png', 'svg')
MAX_ITENS = 14                                             # itens por compartimento antes de resumir em "+ N"
FRACAO_CONFIG = 0.5                                        # variável lida por >= 50% das células = configuração global
FONTE = 'Helvetica'

if not (NOTEBOOK_PATH and os.path.isfile(NOTEBOOK_PATH)):
  for cand in (Path('/content/drive/MyDrive/Colab Notebooks') / NOTEBOOK_NAME, Path('/content/drive/MyDrive') / NOTEBOOK_NAME):
    if cand.is_file():
      NOTEBOOK_PATH = str(cand)
      break
if not (NOTEBOOK_PATH and os.path.isfile(NOTEBOOK_PATH)):
  raise FileNotFoundError('Notebook salvo não encontrado: salve com Ctrl+S ou defina DETECTED_NB_PATH.')

# ------------------------------------------------------------------- 2. Graphviz
if not shutil.which('dot'):
  print('Instalando o Graphviz ...')
  subprocess.run('apt-get -qq install -y graphviz > /dev/null 2>&1', shell=True)
TEM_DOT = bool(shutil.which('dot'))
if not TEM_DOT:
  print('  [aviso] Graphviz indisponível: os diagramas serão gravados só como .dot (texto).')

if UML_DIR.is_dir():                                       # cada execução regrava o conjunto inteiro
  shutil.rmtree(UML_DIR)
(UML_DIR / 'celulas').mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------- 3. leitura das células
_ROTULO = re.compile(r'^#\s*(?:@title\s*#\s*)?C[ée]lula\s+([^:\n]+)', re.M)
fontes = [''.join(c.get('source', [])) for c in json.loads(Path(NOTEBOOK_PATH).read_text(encoding='utf-8'))['cells']
          if c.get('cell_type') == 'code']
celulas = {}                                               # rótulo -> (posição, código); vale a última versão
for pos, src in enumerate(fontes):
  m = _ROTULO.search(src or '')
  if m:
    celulas[m.group(1).strip()] = (pos, src)
ordem = sorted(celulas, key=lambda r: celulas[r][0])


def slug(rotulo: str) -> str:
  return re.sub(r'[^0-9A-Za-z]+', '_', unicodedata.normalize('NFKD', rotulo.lower())
                .encode('ascii', 'ignore').decode()).strip('_')[:40].rstrip('_')


def sem_magicas(src: str) -> str:
  """Converte comandos do Colab (!pip, %cd, x = !ls) em Python puro, como o próprio IPython faz."""
  try:
    from IPython.core.inputtransformer2 import TransformerManager
    return TransformerManager().transform_cell(src)
  except Exception:
    pass
  linhas, aceitas = [], []                                 # alternativa: comando só fora de parênteses e de textos

  def dentro_de_expressao(texto: str) -> bool:
    prof = 0
    try:
      for tok in tokenize.generate_tokens(io.StringIO(texto).readline):
        if tok.type == tokenize.OP and tok.string in '([{':
          prof += 1
        elif tok.type == tokenize.OP and tok.string in ')]}':
          prof -= 1
    except tokenize.TokenError as e:
      return 'string' in str(e) or prof > 0
    except (IndentationError, SyntaxError):
      pass
    return prof > 0

  for ln in src.split('\n'):
    t = ln.lstrip()
    if (t.startswith(('!', '%')) or re.match(r'^[\w\s,]+=\s*!', t)) and not dentro_de_expressao('\n'.join(aceitas) + '\n'):
      linhas.append(ln[:len(ln) - len(t)] + 'pass  # ' + t)
    else:
      linhas.append(ln)
      aceitas.append(ln)
  return '\n'.join(linhas)


# ------------------------------------------------------------------- 4. análise de cada célula (ast)
BUILTINS = set(dir(builtins)) | {'get_ipython', 'display', '__file__', '__name__'}
_ARQ_NOME = re.compile(r'^[\w\-.{}%*]{3,80}\.(csv|png|pdf|json|npz|npy|pt|pth|ckpt|md|txt|zip|html|nc4|h5)$', re.I)


def assinatura(fn) -> str:
  a = [x.arg for x in fn.args.posonlyargs + fn.args.args if x.arg not in ('self', 'cls')]
  if fn.args.vararg:
    a.append('*' + fn.args.vararg.arg)
  a += [x.arg for x in fn.args.kwonlyargs]
  if fn.args.kwarg:
    a.append('**' + fn.args.kwarg.arg)
  txt = ', '.join(a)
  return f"{fn.name}({txt if len(txt) <= 38 else txt[:35] + '…'})"


def nome_base(b) -> str:
  try:
    return ast.unparse(b)
  except Exception:
    return '?'


def alvos(no):
  if isinstance(no, ast.Name):
    yield no.id
  elif isinstance(no, (ast.Tuple, ast.List)):
    for e in no.elts:
      yield from alvos(e)
  elif isinstance(no, ast.Starred):
    yield from alvos(no.value)


def releitura(valor):
  """Devolve 'X' se valor for globals().get('X', ...) ou globals()['X'], senão None."""
  if isinstance(valor, ast.Call) and isinstance(valor.func, ast.Attribute) and valor.func.attr == 'get' \
      and isinstance(valor.func.value, ast.Call) and getattr(valor.func.value.func, 'id', '') == 'globals' \
      and valor.args and isinstance(valor.args[0], ast.Constant):
    return valor.args[0].value
  return None


def analisar(src: str) -> dict:
  info = {'linhas': src.count('\n') + 1, 'imports': set(), 'classes': [], 'funcoes': [], 'produz': set(),
          'le': set(), 'arquivos': set(), 'chamadas': defaultdict(set), 'erro': None}
  try:
    arv = ast.parse(sem_magicas(src))
  except SyntaxError as e:
    info['erro'] = f'linha {e.lineno}: {e.msg}'
    return info
  locais = set()
  for no in ast.walk(arv):
    if isinstance(no, ast.Import):
      for a in no.names:
        info['imports'].add(a.name.split('.')[0])
        locais.add((a.asname or a.name).split('.')[0])
    elif isinstance(no, ast.ImportFrom) and no.module and not no.level:
      info['imports'].add(no.module.split('.')[0])
      locais.update(a.asname or a.name for a in no.names)
    elif isinstance(no, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
      args = no.args
      locais.update(x.arg for x in args.posonlyargs + args.args + args.kwonlyargs)
      locais.update(x.arg for x in (args.vararg, args.kwarg) if x)
    elif isinstance(no, ast.comprehension):
      locais.update(alvos(no.target))
    elif isinstance(no, ast.ExceptHandler) and no.name:
      locais.add(no.name)
    elif isinstance(no, ast.Constant) and isinstance(no.value, str) and _ARQ_NOME.match(os.path.basename(no.value.strip())):
      info['arquivos'].add(os.path.basename(no.value.strip()))
    elif isinstance(no, ast.Call) and isinstance(no.func, ast.Attribute) and no.func.attr == 'get' \
        and isinstance(no.func.value, ast.Call) and getattr(no.func.value.func, 'id', '') == 'globals' \
        and no.args and isinstance(no.args[0], ast.Constant) and isinstance(no.args[0].value, str):
      info['le'].add(no.args[0].value)                      # globals().get('NOME', ...)
  # nomes produzidos no nível da célula (fora de funções e classes)
  def nivel_superior(corpo):
    for st in corpo:
      if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
        info['funcoes'].append(st)
        info['produz'].add(st.name)
      elif isinstance(st, ast.ClassDef):
        info['classes'].append(st)
        info['produz'].add(st.name)
      else:
        for no in ast.walk(st):
          if isinstance(no, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for t in (no.targets if isinstance(no, ast.Assign) else [no.target]):
              nomes = set(alvos(t))
              if releitura(no.value) in nomes:                # X = globals().get('X', ...) só relê X
                nomes.discard(releitura(no.value))
              info['produz'].update(nomes)
          elif isinstance(no, (ast.For, ast.AsyncFor, ast.withitem)):
            alvo = no.target if isinstance(no, (ast.For, ast.AsyncFor)) else no.optional_vars
            if alvo is not None:
              locais.update(alvos(alvo))                    # variáveis de laço não contam como produto
          elif isinstance(no, ast.Global):
            info['produz'].update(no.names)
          elif isinstance(no, ast.Subscript) and isinstance(no.ctx, ast.Store) \
              and isinstance(no.value, ast.Call) and getattr(no.value.func, 'id', '') == 'globals' \
              and isinstance(no.slice, ast.Constant):
            info['produz'].add(str(no.slice.value))         # globals()['NOME'] = ...
  nivel_superior(arv.body)
  for fn in info['funcoes']:                                # funções que declaram global também produzem
    for no in ast.walk(fn):
      if isinstance(no, ast.Global):
        info['produz'].update(no.names)
  # nomes lidos que vêm de fora da célula
  atribuidos = set(info['produz']) | locais
  for no in ast.walk(arv):
    if isinstance(no, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
      for sub in ast.walk(no):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
          atribuidos.add(sub.id)
  for no in ast.walk(arv):
    if isinstance(no, ast.Name) and isinstance(no.ctx, ast.Load) and no.id not in atribuidos and no.id not in BUILTINS:
      info['le'].add(no.id)
  info['le'] -= BUILTINS
  # chamadas entre funções da própria célula e para nomes externos
  nomes_fn = {f.name for f in info['funcoes']}
  for fn in info['funcoes']:
    for no in ast.walk(fn):
      if isinstance(no, ast.Call) and isinstance(no.func, ast.Name) and no.func.id != fn.name:
        info['chamadas'][fn.name].add(no.func.id)
  info['nomes_fn'] = nomes_fn
  return info


analises = {r: analisar(celulas[r][1]) for r in ordem}

# ------------------------------------------------------------------- 5. fluxo de dados entre células
leitores = Counter()
for r in ordem:
  leitores.update(analises[r]['le'])
CONFIG = {n for n, k in leitores.items() if k >= FRACAO_CONFIG * len(ordem)}
arestas = defaultdict(set)                                   # (produtora, consumidora) -> nomes
ultima_produtora, primeira_produtora = {}, {}
for r in ordem:
  a = analises[r]
  for n in a['le']:
    if n in ultima_produtora and ultima_produtora[n] != r:
      arestas[(ultima_produtora[n], r)].add(n)
  for n in a['produz']:
    ultima_produtora[n] = r
    primeira_produtora.setdefault(n, r)
config_origem = {n: primeira_produtora[n] for n in sorted(CONFIG) if n in primeira_produtora}


# ------------------------------------------------------------------- 6. desenho (DOT -> PNG/SVG)
def esc(s) -> str:
  return html.escape(str(s), quote=True)


def compartimento(itens, vazio='—') -> str:
  itens = list(itens)
  if not itens:
    return f'<TR><TD ALIGN="LEFT"><FONT COLOR="#888888">{vazio}</FONT></TD></TR>'
  extra = len(itens) - MAX_ITENS
  linhas = ''.join(f'<TR><TD ALIGN="LEFT">{esc(i)}</TD></TR>' for i in itens[:MAX_ITENS])
  if extra > 0:
    linhas += f'<TR><TD ALIGN="LEFT"><I>+ {extra} mais</I></TD></TR>'
  return linhas


def caixa_uml(estereotipo, titulo, atributos, operacoes, cor='#FFF8E1') -> str:
  return (f'<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="3" BGCOLOR="{cor}">'
          f'<TR><TD><FONT POINT-SIZE="9">«{esc(estereotipo)}»</FONT><BR/><B>{esc(titulo)}</B></TD></TR><HR/>'
          f'<TR><TD><TABLE BORDER="0" CELLSPACING="0" CELLPADDING="1">{compartimento(atributos)}</TABLE></TD></TR><HR/>'
          f'<TR><TD><TABLE BORDER="0" CELLSPACING="0" CELLPADDING="1">{compartimento(operacoes)}</TABLE></TD></TR></TABLE>>')


def renderizar(dot_txt: str, base: Path) -> list:
  base.with_suffix('.dot').write_text(dot_txt, encoding='utf-8')
  feitos = [base.with_suffix('.dot')]
  if TEM_DOT:
    for fmt in FORMATOS:
      out = base.with_suffix('.' + fmt)
      r = subprocess.run(['dot', f'-T{fmt}', '-Gdpi=150' if fmt == 'png' else '-Gdpi=72', '-o', str(out),
                          str(base.with_suffix('.dot'))], capture_output=True, text=True)
      if r.returncode == 0:
        feitos.append(out)
      else:
        print(f'  [aviso] {base.name}.{fmt}: {r.stderr.strip()[:150]}')
  return feitos


CABECALHO = (f'  graph [fontname="{FONTE}", rankdir=LR, nodesep=0.35, ranksep=0.6, labelloc=t, fontsize=14];\n'
             f'  node  [fontname="{FONTE}", fontsize=10, shape=plain];\n'
             f'  edge  [fontname="{FONTE}", fontsize=8, color="#555555"];\n')


def diagrama_celula(rotulo: str, a: dict) -> str:
  n = slug(rotulo)
  publicos = sorted((x for x in a['produz'] if not x.startswith('_') and x not in a['nomes_fn']
                     and x not in {c.name for c in a['classes']}), key=lambda x: (not x.isupper(), x.lower()))
  ops = [('- ' if f.name.startswith('_') else '+ ') + assinatura(f) for f in a['funcoes']]
  t = [f'digraph "{n}" {{\n', CABECALHO,
       f'  label=<<B>Célula {esc(rotulo)}</B> — {a["linhas"]} linhas>;\n',
       f'  celula [label={caixa_uml("célula", "Célula " + rotulo, ["+ " + p for p in publicos], ops)}];\n']
  for c in a['classes']:
    attrs, metodos = set(), []
    for st in c.body:
      if isinstance(st, (ast.Assign, ast.AnnAssign)):
        for alvo in (st.targets if isinstance(st, ast.Assign) else [st.target]):
          attrs.update(alvos(alvo))
      elif isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
        metodos.append(('- ' if st.name.startswith('_') and not st.name.startswith('__') else '+ ') + assinatura(st))
        for no in ast.walk(st):
          if isinstance(no, ast.Attribute) and isinstance(no.ctx, ast.Store) and getattr(no.value, 'id', '') == 'self':
            attrs.add(no.attr)
    cid = 'cls_' + c.name
    t.append(f'  {cid} [label={caixa_uml("classe", c.name, sorted(("- " if x.startswith("_") else "+ ") + x for x in attrs), metodos, "#E3F2FD")}];\n')
    t.append(f'  celula -> {cid} [arrowtail=diamond, dir=back, label="define"];\n')
    for b in c.bases:
      bn = nome_base(b)
      bid = 'base_' + re.sub(r'\W', '_', bn)
      if bn not in {x.name for x in a['classes']}:
        t.append(f'  {bid} [shape=box, style="rounded,dashed", fontsize=9, label="{esc(bn)}"];\n')
      else:
        bid = 'cls_' + bn
      t.append(f'  {cid} -> {bid} [arrowhead=empty, label="herda"];\n')
  libs = sorted(a['imports'])
  if libs:
    rotulos_libs = '<BR/>'.join(esc(x) for x in libs[:MAX_ITENS]) + (f'<BR/><I>+ {len(libs) - MAX_ITENS} mais</I>' if len(libs) > MAX_ITENS else '')
    t.append(f'  libs [shape=tab, style=filled, fillcolor="#F1F8E9", fontsize=9, label=<<B>bibliotecas</B><BR/>{rotulos_libs}>];\n')
    t.append('  celula -> libs [style=dashed, arrowhead=open, label="«import»"];\n')
  externos = sorted(x for x in a['le'] if not x.startswith('_'))
  if externos:
    rot = '<BR/>'.join(esc(x) for x in externos[:MAX_ITENS]) + (f'<BR/><I>+ {len(externos) - MAX_ITENS} mais</I>' if len(externos) > MAX_ITENS else '')
    t.append(f'  entradas [shape=note, style=filled, fillcolor="#FCE4EC", fontsize=9, label=<<B>usa de outras células</B><BR/>{rot}>];\n')
    t.append('  entradas -> celula [style=dashed, arrowhead=open, label="«use»"];\n')
  arqs = sorted(a['arquivos'])
  if arqs:
    rot = '<BR/>'.join(esc(x) for x in arqs[:8]) + (f'<BR/><I>+ {len(arqs) - 8} mais</I>' if len(arqs) > 8 else '')
    t.append(f'  arquivos [shape=folder, style=filled, fillcolor="#EDE7F6", fontsize=9, label=<<B>arquivos citados</B><BR/>{rot}>];\n')
    t.append('  celula -> arquivos [style=dotted, arrowhead=open, label="lê/grava"];\n')
  t.append('}\n')
  return ''.join(t)


def diagrama_fluxo() -> str:
  t = ['digraph pipeline {\n', CABECALHO.replace('rankdir=LR', 'rankdir=TB'),
       '  label=<<B>F-PluvIA Metropolitan — fluxo de dados entre as células</B>>;\n']
  for r in ordem:
    a = analises[r]
    resumo = f"{a['linhas']} linhas · {len(a['classes'])} classes · {len(a['funcoes'])} funções"
    t.append(f'  "{slug(r)}" [shape=component, style=filled, fillcolor="{"#E3F2FD" if a["classes"] else "#FFF8E1"}", '
             f'label=<<B>Célula {esc(r)}</B><BR/><FONT POINT-SIZE="8">{esc(resumo)}</FONT>>];\n')
  for (p, c), nomes in sorted(arestas.items(), key=lambda kv: (celulas[kv[0][0]][0], celulas[kv[0][1]][0])):
    nomes = sorted(nomes - CONFIG)
    if not nomes:
      continue
    rot = ', '.join(nomes[:2]) + (f' +{len(nomes) - 2}' if len(nomes) > 2 else '')
    t.append(f'  "{slug(p)}" -> "{slug(c)}" [label="{esc(rot)}"];\n')
  if config_origem:
    cfg = '<BR/>'.join(esc(f'{n} (Célula {o})') for n, o in list(config_origem.items())[:MAX_ITENS])
    t.append(f'  config [shape=note, style=filled, fillcolor="#FCE4EC", fontsize=9, '
             f'label=<<B>configuração usada por quase todas as células</B><BR/>{cfg}>];\n')
  t.append('}\n')
  return ''.join(t)


# ------------------------------------------------------------------- 7. geração
print(f'Notebook: {Path(NOTEBOOK_PATH).name} — {len(fontes)} células de código, {len(ordem)} com rótulo "Célula ..."')
gerados, falhas, linhas_indice = [], [], []
gerados += renderizar(diagrama_fluxo(), UML_DIR / '00_fluxo_entre_celulas')
for i, r in enumerate(ordem, 1):
  a = analises[r]
  if a['erro']:
    falhas.append((r, a['erro']))
    continue
  base = UML_DIR / 'celulas' / f'{i:02d}_celula_{slug(r)}'
  gerados += renderizar(diagrama_celula(r, a), base)
  linhas_indice.append(f"| {i:02d} | Célula {r} | {a['linhas']} | {len(a['classes'])} | {len(a['funcoes'])} | "
                       f"{len(a['imports'])} | [PNG](celulas/{base.name}.png) · [SVG](celulas/{base.name}.svg) |")

indice = [
  '# Diagramas UML do F-PluvIA Metropolitan', '',
  'Gerados automaticamente pela **Célula Full UMLs** a partir do código salvo de cada célula do notebook.', '',
  '## Fluxo de dados entre as células', '',
  '![Fluxo entre células](00_fluxo_entre_celulas.png)', '',
  'Cada caixa é uma célula (azul: define classes). Uma seta indica que a célula de destino usa variáveis criadas na de '
  'origem; o rótulo mostra até duas delas. Variáveis de configuração lidas por quase todas as células ficam na nota à parte.', '',
  '## Diagramas por célula', '',
  '| # | Célula | Linhas | Classes | Funções | Bibliotecas | Diagrama |', '|---|---|---|---|---|---|---|',
  *linhas_indice, '',
  '## Como ler', '',
  '- **«célula»**: no 1º compartimento, as variáveis que a célula deixa disponíveis para as seguintes; no 2º, as funções que ela define (operações).',
  '- **«classe»**: atributos e métodos (+ público, - interno). Losango: a célula define a classe. Triângulo vazado: herança.',
  '- **bibliotecas** («import»), **usa de outras células** («use») e **arquivos citados** completam o contexto.',
  '- Diagramas de sequência e de atividade não são gerados, porque dependeriam de rastrear a execução.', '']
if falhas:
  indice += ['## Células não analisadas', ''] + [f'- Célula {r}: {e}' for r, e in falhas] + ['']
(UML_DIR / 'README.md').write_text('\n'.join(indice), encoding='utf-8')

print(f'\nDiagramas: {len(ordem) - len(falhas)} células + fluxo entre células')
print(f'Arquivos gravados: {len(gerados) + 1} em {UML_DIR}')
print(f'Variáveis de configuração (nota à parte): {", ".join(sorted(CONFIG)) or "—"}')
for r, e in falhas:
  print(f'  [aviso] Célula {r} não analisada ({e})')
globals()['UML_DIR'] = str(UML_DIR)
