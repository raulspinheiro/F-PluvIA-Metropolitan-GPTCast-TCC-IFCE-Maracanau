# @title
# Célula 13.3: Volume de Tokens por Etapa e por Variante — quantos tokens o Transformer do GPTCast-8x8 processou
# no ajuste fino, na validação e nas previsões de cada variante do F-PluvIA e do Zero-Shot, e quantos tokens custa
# prever um único quadro. Não roda nenhum modelo: usa a arquitetura do GPTCast e os registros já gravados no Drive.
#
# "Token" aqui é um índice do codebook da VQGAN (um bloco de 16 x 16 pixels, ~32 km):
# é o volume de trabalho do Transformer. Contagens conferidas no código do pacote gptcast:
#   - treino/validação: cada amostra tem 8 quadros x 64 tokens = 512; o Transformer recebe o token inicial (SOS)
#     + 511 tokens e é avaliado nos 512 tokens da sequência (shared_step);
#   - previsão: cada quadro futuro tem 64 tokens gerados um a um (predict_next_frame_indices), cada um com uma
#     passada completa do Transformer sobre SOS + 7 quadros de contexto (448 tokens) + os já gerados no quadro.
#     Não há cache de atenção (KV cache, ele evitaria processamento repetido dos mesmos tokens ): o contexto é reprocessado a cada token.
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ENV_TAG = globals().get('ENV_TAG', 'G4_RTX6000_Blackwell_95GB')
ENV_LABEL = globals().get('ENV_LABEL', 'Hardware Ativo')
SERIES_CACHE_DIR = globals().get('SERIES_CACHE_DIR', '/content/drive/MyDrive/GPTCast_RMF/series_2019_2025_2km')
RESULTS_ENV_DIR = globals().get('RESULTS_ENV_DIR', os.path.join(SERIES_CACHE_DIR, f'resultados_{ENV_TAG}'))

# ------------------------------------------------------------- 1. arquitetura e constantes
N_CTX = int(globals().get('CONTEXT_STEPS', 7))          # quadros de contexto na previsão
N_FUT = int(globals().get('FORECAST_STEPS', 4))         # quadros previstos (+30 a +120 min)
BLOCK = 512                                             # block_size do Transformer (8 quadros x 64 tokens)
TOK_QUADRO = 64                                         # 128 x 128 px / (16 x 16 px por token) = 8 x 8
_gpt = globals().get('gpt8')
if _gpt is not None:
  try:
    BLOCK = int(_gpt.transformer.config.block_size)
    TOK_QUADRO = BLOCK // 8
  except Exception:
    pass
TOK_AMOSTRA = BLOCK                                     # tokens da sequência por amostra de treino/validação
N_MEMBROS = 8                                           # F-PluvIA-C8 (média de 8 sorteios)
N_JANELAS_45 = 240                                      # Célula 4.5: janelas da validação de 2024 sorteadas
CTX_INI = 1 + N_CTX * TOK_QUADRO                        # SOS + contexto = 449 tokens no 1º token do quadro
POS_QUADRO = sum(CTX_INI + k for k in range(TOK_QUADRO))   # posições processadas para gerar 1 quadro (30.752)
assert CTX_INI + TOK_QUADRO - 1 <= BLOCK, 'contexto da previsão maior que o block_size'


def _achar(padrao):
  """Arquivo mais recente que casa com o padrão, em qualquer subpasta de RESULTS_ENV_DIR (ou None)."""
  achados = sorted(glob.glob(os.path.join(RESULTS_ENV_DIR, '**', padrao), recursive=True), key=os.path.getmtime)
  return achados[-1] if achados else None


fontes = {}
# ------------------------------------------------------------- 2. tamanhos das partições
N_TREINO, N_VAL, N_TESTE25 = 9940, 2224, 1729
f = _achar(f'resumo_particoes_{ENV_TAG}_*.csv')
if f:
  d = pd.read_csv(f).set_index('Particao')['Sequencias']
  N_TREINO, N_VAL, N_TESTE25 = int(d['Treino']), int(d['Validação']), int(d['Teste'])
  fontes['partições'] = os.path.basename(f)
reg = {}
f = _achar(f'log_tempos_treino_e_inferencia_{ENV_TAG}_*.json')
if f:
  reg = json.load(open(f, encoding='utf-8'))
  fontes['registro de treino'] = os.path.basename(f)
LOTE = int(reg.get('batch_size_treino', 16))
T_AJUSTE_S = float(reg.get('finetuning_tempo_total_s', np.nan))
PASSOS_EPOCA = -(-N_TREINO // LOTE)                     # 9.940 / 16 -> 622 passos (o último com 4 amostras)


def _amostras_e_validacoes(metrics_csv):
  """Amostras de treino vistas e validações completas, lidas do metrics.csv do Lightning."""
  m = pd.read_csv(metrics_csv)
  passos = int(m['step'].max()) + 1
  epocas_completas, resto = divmod(passos, PASSOS_EPOCA)
  amostras = epocas_completas * N_TREINO + min(resto * LOTE, N_TREINO)
  n_val = int(m['val/loss_epoch'].notna().sum()) if 'val/loss_epoch' in m else 0
  return amostras, n_val, passos


# ------------------------------------------------------------- 3. treino e validação
linhas = []


def add(etapa, modelo, variante, unidade, n, tok_unidade, gerados=False, obs=''):
  linhas.append({'Etapa': etapa, 'Modelo': modelo, 'Variante': variante, 'Unidade': unidade, 'Quantidade': int(n),
                 'Tokens por unidade': int(tok_unidade), 'Tokens': int(n * tok_unidade),
                 'Posições processadas pelo Transformer': int(n * tok_unidade * (POS_QUADRO / TOK_QUADRO) if gerados
                                                              else n * tok_unidade),
                 'Tipo': 'gerados (previsão)' if gerados else 'lidos (treino/validação)', 'Observação': obs})


tag = str(reg.get('finetuning_origem', '')).replace('treino de ', '').strip()
principais = [p for p in glob.glob(os.path.join(RESULTS_ENV_DIR, '**', 'metrics.csv'), recursive=True)
              if 'alternativo' not in p and (not tag or tag in p)]
if principais:
  amostras, n_val, passos = _amostras_e_validacoes(sorted(principais)[-1])
  fontes['curva do ajuste fino'] = os.path.relpath(sorted(principais)[-1], RESULTS_ENV_DIR)
else:
  amostras, n_val, passos = int(reg.get('epocas_executadas', 5)) * N_TREINO, int(reg.get('epocas_executadas', 5)), None
add('Ajuste fino (Célula 4)', 'F-PluvIA (lr 5e-5)', 'treino', 'amostra de 8 quadros', amostras, TOK_AMOSTRA,
    obs=f'{amostras // N_TREINO} épocas de {N_TREINO} janelas (recorte aleatório)')
add('Ajuste fino (Célula 4)', 'F-PluvIA (lr 5e-5)', 'validação a cada época', 'amostra de 8 quadros', n_val * N_VAL,
    TOK_AMOSTRA, obs=f'{n_val} validações completas de {N_VAL} janelas de 2024')

for p in sorted(glob.glob(os.path.join(RESULTS_ENV_DIR, '**', 'curva_treino_alternativo_*', '*', 'metrics.csv'),
                          recursive=True)):
  a, v, ps = _amostras_e_validacoes(p)
  rodada = os.path.basename(os.path.dirname(p))
  add('Ajuste fino suave (Célula 4.3)', 'F-PluvIA suave (lr 1e-5)', 'treino', 'amostra de 8 quadros', a, TOK_AMOSTRA,
      obs=f'rodada {rodada[:14]}: {ps} passos' + (' (interrompida no início)' if ps < PASSOS_EPOCA else ''))
  if v:
    add('Ajuste fino suave (Célula 4.3)', 'F-PluvIA suave (lr 1e-5)', 'validação a cada ¼ de época',
        'amostra de 8 quadros', v * N_VAL, TOK_AMOSTRA, obs=f'rodada {rodada[:14]}: {v} validações completas')
  fontes.setdefault('curvas da Célula 4.3', []).append(rodada)

add('Validação Fine-Tuned × Zero-Shot (Célula 4.1)', 'F-PluvIA (lr 5e-5)', 'perda nos tokens', 'amostra de 8 quadros',
    N_VAL, TOK_AMOSTRA, obs='uma passada sobre a validação de 2024')
add('Validação Fine-Tuned × Zero-Shot (Célula 4.1)', 'Zero-Shot', 'perda nos tokens', 'amostra de 8 quadros', N_VAL,
    TOK_AMOSTRA, obs='uma passada sobre a validação de 2024')

# ------------------------------------------------------------- 4. previsões (tokens gerados)
TOK_JANELA = N_FUT * TOK_QUADRO                          # 4 quadros x 64 = 256 tokens gerados por janela
n26, n_casos26 = 2108, 2
f = _achar(f'teste_cego_2026_meses_e_produtos_{ENV_TAG}*.csv')
if f:
  n26 = int(pd.read_csv(f)['Janelas'].sum())
  fontes['teste cego 2026'] = os.path.basename(f)
f = _achar(f'teste_cego_2026_estudos_de_caso_{ENV_TAG}*.csv')
if f:
  n_casos26 = int(pd.read_csv(f)['Caso'].nunique())

DEC_45 = [('Padrão (T=1, sem top-k)', 1), ('top-k 50', 1), ('top-k 10', 1), ('Temperatura 0,7', 1),
          ('Gulosa (top-k 1)', 1), (f'Média de {N_MEMBROS} sorteios', N_MEMBROS)]
for modelo in ('F-PluvIA (lr 5e-5)', 'Zero-Shot'):
  for nome, membros in DEC_45:
    add('Escolha da decodificação (Célula 4.5, validação 2024)', modelo, nome, 'janela (4 quadros)',
        N_JANELAS_45 * membros, TOK_JANELA, gerados=True,
        obs=f'{N_JANELAS_45} janelas' + (f' x {membros} membros' if membros > 1 else ''))

PREV = [  # (etapa, modelo, variante, janelas, membros, observação)
    ('Teste 2025 (Célula 12)', 'F-PluvIA (lr 5e-5)', 'F-PluvIA-A (sorteio padrão)', N_TESTE25, 1, 'rodada com a decodificação padrão do pacote'),
    ('Teste 2025 (Célula 12)', 'Zero-Shot', 'sorteio padrão', N_TESTE25, 1, 'rodada com a decodificação padrão do pacote'),
    ('Teste 2025 (Célula 12)', 'F-PluvIA (lr 5e-5)', 'F-PluvIA-G (gulosa)', N_TESTE25, 1, 'decodificação escolhida em 2024'),
    ('Teste 2025 (Célula 12)', 'Zero-Shot', 'top-k 10', N_TESTE25, 1, 'decodificação escolhida em 2024'),
    ('Teste 2025 (Célula 12)', 'F-PluvIA (lr 5e-5)', f'F-PluvIA-C{N_MEMBROS} (média de {N_MEMBROS})', N_TESTE25, N_MEMBROS, f'{N_MEMBROS} membros por janela'),
    ('Teste cego 2026 (Célula 13.1)', 'F-PluvIA (lr 5e-5)', 'F-PluvIA-G (gulosa)', n26 + n_casos26, 1, f'{n26} janelas da série + {n_casos26} estudos de caso'),
    ('Teste cego 2026 (Célula 13.1)', 'Zero-Shot', 'top-k 10', n26 + n_casos26, 1, f'{n26} janelas da série + {n_casos26} estudos de caso'),
    ('Teste cego 2026 (Célula 13.1)', 'F-PluvIA (lr 5e-5)', f'F-PluvIA-C{N_MEMBROS} (média de {N_MEMBROS})', n26 + n_casos26, N_MEMBROS, f'{N_MEMBROS} membros por janela'),
]
for etapa, modelo, variante, n, membros, obs in PREV:
  add(etapa, modelo, variante, 'janela (4 quadros)', n * membros, TOK_JANELA, gerados=True, obs=obs)

df = pd.DataFrame(linhas)

# ------------------------------------------------------------- 5. custo de um quadro
vel = {}
f = _achar(f'velocidade_inferencia_teste2025_{ENV_TAG}_N*_amostragem2024_media8.csv')
if f:
  v = pd.read_csv(f).set_index('Modelo')['Tempo_Medio_4Passos_s']
  vel = {'G': v.get('GPTCast-8x8 (Fine-Tuned RMF)'), 'C8': v.get('GPTCast-8x8 (Fine-Tuned RMF, média de 8)'),
         'ZS': v.get('GPTCast-8x8 (Zero-Shot)')}
  fontes['tempos de previsão'] = os.path.basename(f)
unid = []
for nome, membros, chave in (('F-PluvIA-A / F-PluvIA-G / Zero-Shot (1 sorteio)', 1, 'G'),
                             (f'F-PluvIA-C{N_MEMBROS} (média de {N_MEMBROS})', N_MEMBROS, 'C8')):
  t_quadro = vel.get(chave) / N_FUT if vel.get(chave) else np.nan
  unid.append({'Previsão de 1 quadro (+30 min)': nome, 'Tokens gerados': TOK_QUADRO * membros,
               'Passadas do Transformer': TOK_QUADRO, 'Contexto por passada [tokens]': f'{CTX_INI} a {CTX_INI + TOK_QUADRO - 1}',
               'Posições processadas': POS_QUADRO * membros, 'Tempo medido por quadro [s]': round(t_quadro, 3),
               'Tokens gerados por segundo': round(TOK_QUADRO * membros / t_quadro, 1) if t_quadro == t_quadro else np.nan})
df_unid = pd.DataFrame(unid)

tok_ajuste = df[(df['Etapa'] == 'Ajuste fino (Célula 4)')]['Tokens'].sum()
vazao_treino = tok_ajuste / T_AJUSTE_S if T_AJUSTE_S == T_AJUSTE_S and T_AJUSTE_S > 0 else np.nan

# ------------------------------------------------------------- 6. saídas
fmt = lambda x: f'{x:,.0f}'.replace(',', '.')
print(f'=== Volume de tokens do Transformer do GPTCast-8x8 | {ENV_LABEL} ===')
print(f'Arquitetura: {TOK_QUADRO} tokens por quadro | block_size {BLOCK} | contexto de previsão {N_CTX} quadros | '
      f'{N_FUT} quadros previstos | fontes: {fontes}')


def _bonito(d, inteiros):
  """Cópia só para exibir, com separador de milhar (os CSV guardam os números puros)."""
  d = d.copy()
  for c in inteiros:
    d[c] = d[c].map(fmt)
  return d


print('\n--- Previsão de um único quadro ---')
display(_bonito(df_unid, ['Tokens gerados', 'Passadas do Transformer', 'Posições processadas'])
        .set_index('Previsão de 1 quadro (+30 min)'))
print(f'Para gerar {TOK_QUADRO} tokens, o Transformer processa {fmt(POS_QUADRO)} posições ({POS_QUADRO / TOK_QUADRO:.1f} por '
      'token gerado): sem cache de atenção, o contexto inteiro é relido a cada token.')
if vazao_treino == vazao_treino:
  print(f'No ajuste fino (treino + validação), a vazão foi de ~{fmt(vazao_treino)} tokens/s: o treino lê a sequência '
        'inteira de uma vez, a previsão gera um token por vez.')

cols = ['Quantidade', 'Tokens por unidade', 'Tokens', 'Posições processadas pelo Transformer', 'Observação']
print('\n--- Por etapa e variante ---')
with pd.option_context('display.max_rows', None, 'display.max_colwidth', 70):
  display(_bonito(df, cols[:4]).set_index(['Etapa', 'Modelo', 'Variante'])[cols])

resumo = df.pivot_table(index='Modelo', columns='Tipo', values='Tokens', aggfunc='sum', fill_value=0, sort=False)
resumo['Total de tokens'] = resumo.sum(axis=1)
resumo['Posições processadas'] = df.groupby('Modelo', sort=False)['Posições processadas pelo Transformer'].sum()
print('\n--- Por modelo (todas as etapas) ---')
display(_bonito(resumo, list(resumo.columns)))
tot = df['Tokens'].sum()
print(f'Total no trabalho: {fmt(tot)} tokens ({fmt(df["Posições processadas pelo Transformer"].sum())} posições processadas).')
print('Fora da conta: o primeiro ajuste fino (tokenização quadro a quadro, descartado), as previsões de versões '
      'anteriores, os estudos de caso de 18/03/2024 (1 janela, refeitos a cada sessão) e a checagem de sanidade do Lightning.')

p_det = os.path.join(RESULTS_ENV_DIR, f'tokens_por_etapa_e_variante_{ENV_TAG}.csv')
p_uni = os.path.join(RESULTS_ENV_DIR, f'tokens_por_quadro_{ENV_TAG}.csv')
df.to_csv(p_det, index=False)
df_unid.to_csv(p_uni, index=False)

# figura: tokens por etapa e variante (escala log), cores por modelo
cores = {'F-PluvIA (lr 5e-5)': '#d62728', 'F-PluvIA suave (lr 1e-5)': '#ff9896', 'Zero-Shot': '#ff7f0e'}
g = df.groupby(['Etapa', 'Modelo', 'Variante'], sort=False)['Tokens'].sum().reset_index()
_curto = {'F-PluvIA (lr 5e-5)': 'F-PluvIA', 'F-PluvIA suave (lr 1e-5)': 'F-PluvIA suave', 'Zero-Shot': 'Zero-Shot'}
g['rótulo'] = [f"{e.split(' (')[0]} | " + (v if v.startswith('F-PluvIA') else f"{_curto.get(m, m)}, {v}")
               for e, m, v in zip(g['Etapa'], g['Modelo'], g['Variante'])]
fig, ax = plt.subplots(figsize=(12, 0.32 * len(g) + 1.6), constrained_layout=True)
y = np.arange(len(g))[::-1]
ax.barh(y, g['Tokens'], color=[cores.get(m, '#7f7f7f') for m in g['Modelo']])
for yi, t in zip(y, g['Tokens']):
  ax.text(t * 1.08, yi, fmt(t), va='center', fontsize=7.5)
ax.set_yticks(y)
ax.set_yticklabels(g['rótulo'], fontsize=8)
ax.set_xscale('log')
ax.set_xlim(left=max(1, g['Tokens'].min() / 3), right=g['Tokens'].max() * 6)
ax.set_xlabel('Tokens (escala logarítmica): lidos no treino/validação, gerados na previsão')
ax.grid(True, axis='x', ls=':', alpha=0.6)
ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in cores.values()], list(cores), loc='lower right', fontsize=8)
ax.set_title(f'Volume de tokens do Transformer do GPTCast-8x8 por etapa e variante | {ENV_LABEL}', fontweight='bold')
p_fig = os.path.join(RESULTS_ENV_DIR, f'painel_tokens_por_etapa_{ENV_TAG}.png')
fig.savefig(p_fig, dpi=300, bbox_inches='tight')
plt.show()

CELL13_3_RESULTS = {'por_etapa': df, 'por_quadro': df_unid, 'por_modelo': resumo}
print(f'[Célula 13.3] Tabelas -> {p_det} | {p_uni}\n[Célula 13.3] Figura -> {p_fig}')
