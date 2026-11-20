# @title
# Célula 13.4: Análise de Complexidade do F-PluvIA-G (Big O) — estudo de caso comparativo entre duas implementações
# equivalentes do gerador guloso: recálculo integral (sem cache) × memoização das chaves e valores da atenção (KV cache).
# Metodologia do estudo de caso de Programação Dinâmica (Edit Distance, top-down com memoização × bottom-up) durante a discipina de
# ANALG 2051.1 IFCE Maracanaú Profº Thiago Alves .
# semente 42, k tamanhos sorteados e igualmente espaçados, m instâncias sorteadas e iguais para as duas versões,
# tempo médio por instância com perf_counter e verificação de correção com assert.
import math
import os
import random
import time

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ------------------------------------------------------------------- 1. parâmetros do experimento
ENV_TAG = globals().get('ENV_TAG', 'Ambiente_Ativo')
ENV_LABEL = globals().get('ENV_LABEL', 'Hardware Ativo')
SERIES_CACHE_DIR = globals().get('SERIES_CACHE_DIR', '/content/drive/MyDrive/GPTCast_RMF/series_2019_2025_2km')
RESULTS_ENV_DIR = globals().get('RESULTS_ENV_DIR', os.path.join(SERIES_CACHE_DIR, f'resultados_{ENV_TAG}'))
SEED = 42                           # reprodutibilidade, como no estudo de caso
MIN_S, MAX_S = 1, 12                # intervalo dos tamanhos de entrada: passos de 30 min (+30 a +360 min)
K_RANGE = (6, 10)                   # nº de tamanhos distintos (sorteado)
M_RANGE = (10, 20)                  # nº de instâncias por tamanho (sorteado, o mesmo para todos os tamanhos)
CONTEXTO = 7                        # quadros observados, como na Célula 12
AQUECIMENTO = 2                     # execuções descartadas antes de medir (compilação de kernels na GPU)
TOL_EMPATE = 1e-3                   # diferença entre os 2 maiores logits abaixo da qual há empate numérico

for _nome in ('gpt8', 'md_rmf'):
  if _nome not in globals():
    raise RuntimeError(f'"{_nome}" não está na sessão: rode as Células 1 a 4 e a (Activates Checkpoint) antes.')
DEVICE = next(gpt8.parameters()).device
if DEVICE.type != 'cuda':
  print('[aviso] O modelo está na CPU: os tempos não representam a execução na GPU usada no artigo.')
os.makedirs(RESULTS_ENV_DIR, exist_ok=True)
gpt8.eval()
TR = gpt8.transformer
BLOCK = int(TR.config.block_size)                         # contexto máximo do Transformer (512)
WS = int(math.sqrt(BLOCK // 8))                           # janela espacial em tokens (8 → 8×8 = 64 por quadro)
T_QUADRO = WS * WS
P_TRANSFORMER = sum(p.numel() for p in TR.parameters())
seqs = md_rmf.splits['test'].sequences_dbz


def br(n) -> str:
  return f'{int(n):,}'.replace(',', '.')


def cronometro():
  if DEVICE.type == 'cuda':
    torch.cuda.synchronize()
  return time.perf_counter()


# ------------------------------------------------------------------- 2. as duas implementações
def _proximo_token(logits):
  """Mesma regra de predict_next_index com temperatura 1 e top-k 1 (decodificação gulosa)."""
  lg = gpt8.top_k_logits(logits[:, -1, :] / 1.0, 1)
  _, res = torch.topk(F.softmax(lg, dim=-1), k=1, dim=-1)
  return res


def quadro_recalculo(ind_ctx, c_idx):
  """Versão 1, sem cache: o gerador original do pacote. Cada um dos T tokens do quadro
  reprocessa o contexto inteiro (condicionamento + quadros passados + tokens já gerados)."""
  return gpt8.predict_next_frame_indices(ind_ctx, c_idx, WS, temperature=1.0, top_k=1, show_progress=False)


def quadro_kv_cache(ind_ctx, c_idx):
  """Versão 2, com memoização: o contexto passa uma única vez pelo Transformer e as chaves e valores
  de cada camada ficam guardados (KV cache); cada token novo processa só 1 posição."""
  b, _, h, w = ind_ctx.shape
  if (h, w) != (WS, WS):
    raise RuntimeError(f'O KV cache exige que a janela ({WS}×{WS}) cubra o quadro inteiro; quadro de {h}×{w} tokens.')
  prefixo = torch.cat((c_idx.reshape(b, -1), ind_ctx.reshape(b, -1)), dim=1)
  logits, _, cache = TR.forward_with_past(prefixo)
  n = prefixo.shape[1]
  tok = _proximo_token(logits)
  tokens = [tok]
  for _ in range(h * w - 1):
    logits, _, novo = TR.forward_with_past(tok, past=[cache], past_length=n)
    cache = torch.cat((cache, novo), dim=-2)                # o pacote devolve só k e v da posição nova
    n += 1
    tok = _proximo_token(logits)
    tokens.append(tok)
  return torch.cat(tokens, dim=1).reshape(b, h, w)


GERADORES = {'Recálculo (sem cache)': quadro_recalculo, 'KV cache (memoização)': quadro_kv_cache}


@torch.inference_mode()
def codificar(ctx_dbz):
  x = torch.tensor(np.clip(np.nan_to_num(ctx_dbz, nan=0.0), 0, 60) / 30.0 - 1.0, dtype=torch.float32, device=DEVICE)
  s = x.shape[0]
  xs = einops.rearrange(x, 's h w -> (s h) w')[None, None].contiguous()
  quant, idx = gpt8.encode_to_z(xs)
  _, c_idx = gpt8.encode_to_c(xs)
  qs = quant.shape
  return idx.reshape(qs[0], s, qs[2] // s, qs[3]), c_idx, qs


@torch.inference_mode()
def prever(ctx_dbz, passos: int, gerador):
  """Previsão completa de S passos (a mesma da Célula 12): codifica, gera quadro a quadro com a janela
  deslizante dos últimos CONTEXTO quadros e decodifica cada quadro. Só o gerador de tokens muda."""
  ind, c_idx, qs = codificar(ctx_dbz)
  s, ih, iw = ind.shape[1], ind.shape[2], ind.shape[3]
  quadros = []
  for _ in range(passos):
    with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == 'cuda'):
      p = gerador(ind[:, -s:], c_idx)
    gpt8.decode_to_img(p.reshape(1, -1), torch.Size([1, qs[1], ih, iw]))
    ind = torch.cat((ind, p[:, None]), dim=1)
    quadros.append(p)
  return torch.stack(quadros, dim=1)


@torch.inference_mode()
def verificar_token_a_token(ctx_dbz):
  """Verificação de correção em fp32, sem divergência em cascata: em cada um dos T tokens do quadro, as duas
  versões recebem o MESMO contexto e devem escolher o mesmo token (logits iguais a menos de arredondamento)."""
  ind, c_idx, _ = codificar(ctx_dbz)
  ind_ctx = ind[:, -CONTEXTO:]
  b = ind_ctx.shape[0]
  prefixo = torch.cat((c_idx.reshape(b, -1), ind_ctx.reshape(b, -1)), dim=1)
  logits_kv, _, cache = TR.forward_with_past(prefixo)
  n, gerados, dif_max, divergentes, empates = prefixo.shape[1], [], 0.0, 0, 0
  for t in range(T_QUADRO):
    contexto = torch.cat([prefixo] + gerados, dim=1)
    l_rc = TR(contexto)[0][:, -1, :].float()
    l_kv = logits_kv[:, -1, :].float()
    dif_max = max(dif_max, float((l_rc - l_kv).abs().max()))
    if int(l_rc.argmax(-1)) != int(l_kv.argmax(-1)):
      divergentes += 1
      top2 = l_rc.topk(2, dim=-1).values[0]
      empates += int(float(top2[0] - top2[1]) < TOL_EMPATE)
    tok = _proximo_token(logits_kv)
    gerados.append(tok)
    if t < T_QUADRO - 1:
      logits_kv, _, novo = TR.forward_with_past(tok, past=[cache], past_length=n)
      cache = torch.cat((cache, novo), dim=-2)
      n += 1
  return dif_max, divergentes, empates


# ------------------------------------------------------------------- 3. sorteio de k, m e dos tamanhos
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
k = random.randint(*K_RANGE)                              # sorteia k
m = random.randint(*M_RANGE)                              # sorteia m (o mesmo para todos os tamanhos)
passo = (MAX_S - MIN_S) / (k - 1)                          # tamanhos igualmente espaçados
tamanhos = sorted({int(MIN_S + i * passo + 1e-9) for i in range(k)})
_amostra = np.clip(seqs[0, :CONTEXTO], 0, 60)
P0 = int(codificar(_amostra)[1].numel() + CONTEXTO * T_QUADRO)            # tokens do contexto antes do 1º token
POS_RC = T_QUADRO * P0 + T_QUADRO * (T_QUADRO - 1) // 2   # posições por quadro, recálculo: Σ (P0 + t)
POS_KV = P0 + T_QUADRO - 1                                 # posições por quadro, KV cache: P0 + (T − 1)
print(f'Transformer: {P_TRANSFORMER / 1e6:.2f} M de parâmetros, {TR.config.n_layer} camadas, contexto máximo {BLOCK}, '
      f'{T_QUADRO} tokens por quadro, {P0} tokens de contexto inicial')
print(f'Sorteio (semente {SEED}): k = {k} tamanhos {tamanhos}, m = {m} instâncias por tamanho | '
      f'posições por quadro: {br(POS_RC)} (recálculo) × {br(POS_KV)} (KV cache), {POS_RC / POS_KV:.0f}× menos')

t0_total = time.perf_counter()

# ------------------------------------------------------------------- 4. verificação de correção (fp32, token a token)
_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
janelas_verif = random.sample(range(len(seqs)), m)
verif = []
try:
  for jv in janelas_verif:
    verif.append(verificar_token_a_token(np.clip(seqs[jv, :CONTEXTO], 0, 60)))
finally:
  torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = _tf32
DIF_MAX = max(v[0] for v in verif)
N_DIV = sum(v[1] for v in verif)
N_EMP = sum(v[2] for v in verif)
assert N_DIV == N_EMP, (f'As versões divergem fora de empate numérico em {N_DIV - N_EMP} de {m * T_QUADRO} tokens: '
                        'o KV cache não reproduz o recálculo.')
print(f'Verificação (fp32): {m} instâncias × {T_QUADRO} tokens, mesmo token em {m * T_QUADRO - N_DIV} de '
      f'{m * T_QUADRO} ({N_EMP} empates numéricos), |Δ logit| máx. = {DIF_MAX:.2e}  OK')

# ------------------------------------------------------------------- 5. tempo médio por instância × tamanho
for _ in range(AQUECIMENTO):
  for _g in GERADORES.values():
    prever(_amostra, 1, _g)

linhas = []
for S in tamanhos:
  totais = {nome: 0.0 for nome in GERADORES}
  identicos = 0
  janelas = random.sample(range(len(seqs)), m)            # *uma única* lista de instâncias para este tamanho
  for j in janelas:                                       # as duas versões sobre as MESMAS instâncias
    ctx = np.clip(seqs[j, :CONTEXTO], 0, 60)
    saidas = {}
    for nome, g in GERADORES.items():
      inicio = cronometro()
      saidas[nome] = prever(ctx, S, g)
      totais[nome] += cronometro() - inicio
    identicos += int(torch.equal(*saidas.values()))
  t_rc, t_kv = (totais[nome] / m for nome in GERADORES)
  linhas.append({'Ambiente_Execucao': ENV_LABEL, 'Passos (S)': S, 'Horizonte [min]': 30 * S, 'Instâncias (m)': m,
                 'Tempo médio recálculo [s]': round(t_rc, 4), 'Tempo médio KV cache [s]': round(t_kv, 4),
                 'Speedup (recálculo / KV)': round(t_rc / t_kv, 2),
                 'Posições recálculo': S * POS_RC, 'Posições KV cache': S * POS_KV,
                 'TFLOP/s recálculo': round(2 * P_TRANSFORMER * S * POS_RC / t_rc / 1e12, 1),
                 'TFLOP/s KV cache': round(2 * P_TRANSFORMER * S * POS_KV / t_kv / 1e12, 1),
                 'Previsões idênticas (bf16)': f'{identicos}/{m}'})
  print(f'S = {S:2d} ({30 * S:3d} min)  OK  recálculo = {t_rc:.3f} s  KV cache = {t_kv:.3f} s  '
        f'speedup = {t_rc / t_kv:.2f}×  idênticas = {identicos}/{m}')
df = pd.DataFrame(linhas)

# Ajuste t = a + b·S para cada versão (O(S) nas duas: a janela desliza e o contexto nunca passa de BLOCK)
S_arr = df['Passos (S)'].to_numpy(float)
ajustes = {}
for nome, col in zip(GERADORES, ['Tempo médio recálculo [s]', 'Tempo médio KV cache [s]']):
  y = df[col].to_numpy(float)
  b_, a_ = np.polyfit(S_arr, y, 1)
  r2 = 1 - np.sum((y - (a_ + b_ * S_arr)) ** 2) / np.sum((y - y.mean()) ** 2)
  ajustes[nome] = (a_, b_, r2, np.polyfit(S_arr, y, 2)[0])
csv_path = os.path.join(RESULTS_ENV_DIR, f'tabela_complexidade_bigO_fpluvia_g_{ENV_TAG}.csv')
df.to_csv(csv_path, index=False)
verif_path = os.path.join(RESULTS_ENV_DIR, f'tabela_verificacao_kvcache_fpluvia_g_{ENV_TAG}.csv')
pd.DataFrame([{'Ambiente_Execucao': ENV_LABEL, 'Janela': jv, 'Tokens': T_QUADRO, 'Max |Δ logit|': v[0],
               'Tokens divergentes': v[1], 'Empates numéricos': v[2]}
              for jv, v in zip(janelas_verif, verif)]).to_csv(verif_path, index=False)

# ------------------------------------------------------------------- 6. painel
CORES = {'Recálculo (sem cache)': '#c0392b', 'KV cache (memoização)': '#2471a3'}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
ax = axes[0]
xs_fit = np.linspace(0, max(tamanhos), 50)
for nome, col in zip(GERADORES, ['Tempo médio recálculo [s]', 'Tempo médio KV cache [s]']):
  a_, b_, r2, _ = ajustes[nome]
  ax.plot(S_arr, df[col], 'o-', color=CORES[nome], label=nome)
  ax.plot(xs_fit, a_ + b_ * xs_fit, ':', color=CORES[nome], alpha=0.7,
          label=f'  t ≈ {a_:.2f} + {b_:.3f}·S s (R² = {r2:.4f})')
ax.set_xlabel('Tamanho da entrada: passos de previsão S (× 30 min)')
ax.set_ylabel('Tempo médio por instância [s]')
ax.set_title(f'(a) Comparação de tempo – k = {k}, m = {m}', fontweight='bold')
ax.set_xticks(tamanhos)
ax.grid(True, ls=':', alpha=0.6)
ax.legend(fontsize=8.5)

ax = axes[1]
razao_pos = POS_RC / POS_KV
speedup_med = float(df['Speedup (recálculo / KV)'].mean())
barras = ax.bar(['Posições processadas\n(teórico, Θ)', 'Tempo medido\n(média dos tamanhos)'],
                [razao_pos, speedup_med], color=['#7f8c8d', CORES['KV cache (memoização)']])
for bar_, v in zip(barras, [razao_pos, speedup_med]):
  ax.annotate(f'{v:.1f}×', (bar_.get_x() + bar_.get_width() / 2, v), ha='center', va='bottom',
              fontsize=11, fontweight='bold')
ax.set_yscale('log')
ax.set_ylim(min(0.5, speedup_med / 2), razao_pos * 1.8)
ax.axhline(1, color='#555555', lw=0.8)
ax.set_ylabel('Razão recálculo / KV cache (escala log)')
ax.set_title('(b) Ganho assintótico × ganho real: o peso das constantes', fontweight='bold')
ax.grid(True, axis='y', ls=':', alpha=0.6)
fig.suptitle(f'F-PluvIA-G: recálculo O(S·T·L·(n·d² + n²·d)) × KV cache O(S·L·(n·d² + n²·d)), '
             f'n ≤ {BLOCK}, T = {T_QUADRO} | {ENV_LABEL}', fontsize=11, fontweight='bold')
fig_path = os.path.join(RESULTS_ENV_DIR, f'painel_complexidade_bigO_fpluvia_g_{ENV_TAG}.png')
fig.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------- 7. resultados e discussão
display(df.drop(columns=['Ambiente_Execucao']).set_index('Passos (S)'))
t_total = time.perf_counter() - t0_total
a_rc, b_rc, r2_rc, c2_rc = ajustes['Recálculo (sem cache)']
a_kv, b_kv, r2_kv, c2_kv = ajustes['KV cache (memoização)']
print('\nResumo do estudo de caso (F-PluvIA-G, decodificação gulosa):')
print(f'  • Correção: as duas versões escolhem o mesmo token em todas as posições verificadas fora de empates '
      f'numéricos (|Δ logit| máx. = {DIF_MAX:.1e}).')
print(f'  • Recálculo: t ≈ {a_rc:.3f} + {b_rc:.3f}·S s (R² = {r2_rc:.4f}, termo quadrático {c2_rc:+.4f}).')
print(f'  • KV cache:  t ≈ {a_kv:.3f} + {b_kv:.3f}·S s (R² = {r2_kv:.4f}, termo quadrático {c2_kv:+.4f}).')
print('    As duas são O(S) no horizonte; a diferença está no custo por quadro: Θ(T) passagens de n tokens '
      'no recálculo contra 1 passagem do contexto + (T − 1) passagens de 1 token com o cache.')
print(f'  • Trabalho: {br(POS_RC)} → {br(POS_KV)} posições por quadro ({razao_pos:.0f}× menos); tempo: '
      f'{speedup_med:.2f}× menos em média. A distância entre as duas razões são as constantes: com lote 1, cada '
      'passagem de 1 token é limitada pelo lançamento de kernels e pela leitura dos pesos da memória, não pelos FLOPs.')
print(f'[Célula 13.4] Tabela -> {csv_path}\n[Célula 13.4] Verificação -> {verif_path}\n'
      f'[Célula 13.4] Painel -> {fig_path}\n[Célula 13.4] Tempo total de execução: {t_total:.0f} s')
