# @title
# Célula 12: Teste Independente de 2025 — os métodos REALMENTE executados em todas as janelas
# (N = 1.729), com checkpoint incremental no Drive, métricas agregadas por horizonte, HSS/ETS,
# FSS e SAL em conjunto, e intervalos de confiança por bootstrap em blocos diários.
# Amostragem dos tokens do GPTCast escolhida na validação de 2024 (Célula 4.5): Fine-Tuned gulosa, Zero-Shot
# top-k 10 e, como 6º método, a média de 8 sorteios do Fine-Tuned. AMOSTRAGEM = 'padrao' refaz a execução antiga.
#
import contextlib
import datetime
import glob
import hashlib
import io
import json
import math
import os
import time
import warnings

import einops
import numpy as np
import pandas as pd
import torch
from gptcast.utils.converters import dbz_to_rainfall, rainfall_to_dbz
from skimage.metrics import structural_similarity as ssim

ENV_TAG = globals().get('ENV_TAG', 'Ambiente_Ativo')
ENV_LABEL = globals().get('ENV_LABEL', 'Hardware Ativo')
RESULTS_ENV_DIR = globals().get('RESULTS_ENV_DIR', os.path.join(SERIES_CACHE_DIR, f'resultados_{ENV_TAG}'))
os.makedirs(RESULTS_ENV_DIR, exist_ok=True)

# ------------------------------------------------------------------- 1. configuração
MAX_TEST_SAMPLES = None    # None = partição de teste inteira
SEPARAR_BLOCOS_CPU = True  # blocos antigos que parecem ter sido gerados na CPU são movidos e refeitos na GPU
SEED = 42                  # semente do GPTCast na janela i = SEED + i (reproduzível)
CHUNK = 50                 # janelas por arquivo de checkpoint
THRESHOLDS = (0.5, 1.0, 5.0, 10.0)
N_BOOT = 1000              # reamostragens do bootstrap em blocos diários
SAL_STRIDE = 1             # SAL em 1 de cada SAL_STRIDE janelas (1 = todas)
STEP_MIN = 30

# Amostragem dos tokens do GPTCast: (temperatura, top_k, sorteios). Escolhida na validação de 2024 pela regra da
# Célula 4.5 (maior CSI ≥ 0,5 mm/h com IC 95 % do ganho sobre o padrão todo acima de zero), sem usar 2025.
AMOSTRAGEM = 'escolhida_2024'      # 'escolhida_2024' (atual) ou 'padrao' (forecast() do pacote, execução antiga)
INCLUIR_MEDIA = True               # 6º método: média de 8 sorteios do Fine-Tuned (cerca de 3x o tempo da execução)
AMOSTRAGENS = {
    'padrao': {'ft': (1.0, None, 1), 'zs': (1.0, None, 1)},
    'escolhida_2024': {'ft': (1.0, 1, 1), 'zs': (1.0, 10, 1), 'ft_media': (1.0, None, 8)},
}
CFG = {k: v for k, v in AMOSTRAGENS[AMOSTRAGEM].items() if k != 'ft_media' or INCLUIR_MEDIA}
DESC = {'ft': {(1.0, None, 1): 'padrão (T=1, sem top-k)', (1.0, 1, 1): 'gulosa (top-k 1)'},
        'zs': {(1.0, None, 1): 'padrão (T=1, sem top-k)', (1.0, 10, 1): 'top-k 10'}}
AMOSTRAGEM_TXT = (f"Fine-Tuned {DESC['ft'][CFG['ft']]}, Zero-Shot {DESC['zs'][CFG['zs']]}"
                  + (f", média de {CFG['ft_media'][2]} sorteios do Fine-Tuned" if 'ft_media' in CFG else ''))
METHODS = ['GPTCast-8x8 (Fine-Tuned RMF)', 'GPTCast-8x8 (Zero-Shot)', 'LINDA (pysteps)',
           'Fluxo Óptico LK (pysteps)', 'Persistência (Baseline)']
KEYS = {'GPTCast-8x8 (Fine-Tuned RMF)': 'ft', 'GPTCast-8x8 (Zero-Shot)': 'zs', 'LINDA (pysteps)': 'linda',
        'Fluxo Óptico LK (pysteps)': 'lk', 'Persistência (Baseline)': 'pers'}
NOME_MEDIA = 'GPTCast-8x8 (Fine-Tuned RMF, média de 8)'
if 'ft_media' in CFG:
  METHODS.insert(1, NOME_MEDIA)
  KEYS[NOME_MEDIA] = 'ft_media'

# ------------------------------------------------------------- aceleração na GPU (CUDA)
# Os dois GPTCast precisam estar na GPU: na CPU cada janela leva dezenas de segundos (parecia um laço infinito).
# As opções numéricas são as mesmas da Célula 4, fixadas aqui para que blocos gerados em sessões diferentes usem
# exatamente a mesma configuração. A precisão é a do próprio pacote gptcast (predict_sequence): o Transformer roda em
# bf16 (gpt_precision padrão) e a VQGAN em float32 (ae_precision padrão), como no estudo de caso da Célula 4. A célula
# não muda esses padrões, para não alterar as previsões no meio da avaliação.
if not torch.cuda.is_available():
  raise RuntimeError('A Célula 12 roda o GPTCast em 1.729 janelas: ative a GPU (Ambiente de execução -> GPU).')
DEVICE = 'cuda'
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
for _nome in ('gpt8', 'gpt8_zeroshot'):
  _m = globals()[_nome]
  if next(_m.parameters()).device.type != 'cuda' or getattr(_m, 'device', torch.device('cuda')).type != 'cuda':
    print(f'[Célula 12] Movendo {_nome} para a GPU.')
  globals()[_nome] = _m.to(DEVICE).eval()
gpt8, gpt8_zeroshot = globals()['gpt8'], globals()['gpt8_zeroshot']
GPU_NOME = torch.cuda.get_device_name(0)

test_ds = md_rmf.splits['test']
seqs_dbz = test_ds.sequences_dbz if MAX_TEST_SAMPLES is None else test_ds.sequences_dbz[:MAX_TEST_SAMPLES]
ts_end = test_ds.timestamps[:len(seqs_dbz)]
N_EVAL = len(seqs_dbz)
ts_t0 = ts_end - np.timedelta64(STEP_MIN * FORECAST_STEPS, 'm')

# Cache legado (versão antiga desta célula): os arquivos cache_cell12_full_1729_test2025_*.npz guardavam
# previsões EMULADAS, não geradas pelos modelos de fato. Eles são renomeados (sufixo _INVALIDO_versao_emulada),
# não apagados: ficam no Drive como registro, mas nenhuma célula volta a lê-los. As previsões reais ficam nos
# blocos de cell12_previsoes_<RUN_ID>/. Sem arquivos antigos, o laço não faz nada.
for old in glob.glob(os.path.join(RESULTS_ENV_DIR, 'cache_cell12_full_1729_test2025_*.npz')):
  if '_INVALIDO_' in os.path.basename(old):     # já renomeado numa execução anterior
    continue
  os.rename(old, old.replace('.npz', '_INVALIDO_versao_emulada.npz'))
  print(f'[Célula 12] Cache antigo (previsões emuladas) renomeado: {os.path.basename(old)}')


def _digest(model) -> str:
  h = hashlib.sha256()
  for k, v in sorted(model.state_dict().items()):
    if k.startswith('transformer.'):
      h.update(k.encode())
      h.update(v.detach().float().cpu().numpy().tobytes())
  return h.hexdigest()[:12]


RUN_BASE = f'{_digest(gpt8)}_n{N_EVAL}_s{SEED}'
RUN_A = f'{RUN_BASE}_amostragem2024'
RUN_ID = RUN_BASE if AMOSTRAGEM == 'padrao' else (f'{RUN_A}_media8' if 'ft_media' in CFG else RUN_A)
CKPT_DIR = os.path.join(RESULTS_ENV_DIR, f'cell12_previsoes_{RUN_ID}')
# Reaproveitamento entre execuções (as previsões são as mesmas, então não há motivo para gerar de novo):
#  - LINDA, Lucas-Kanade e persistência são determinísticos e não dependem da amostragem do GPTCast: vêm da pasta da
#    execução padrão, quando existe;
#  - com a média de 8, o Fine-Tuned guloso e o Zero-Shot top-k 10 vêm da pasta sem a média, quando já existe.
BASE_DIR = os.path.join(RESULTS_ENV_DIR, f'cell12_previsoes_{RUN_BASE}')
A_DIR = os.path.join(RESULTS_ENV_DIR, f'cell12_previsoes_{RUN_A}')
REUSO = [] if AMOSTRAGEM == 'padrao' else [(BASE_DIR, ('linda', 'lk', 'pers'))] + \
        ([(A_DIR, ('ft', 'zs'))] if 'ft_media' in CFG else [])
SUF = '' if AMOSTRAGEM == 'padrao' else ('_amostragem2024_media8' if 'ft_media' in CFG else '_amostragem2024')
os.makedirs(CKPT_DIR, exist_ok=True)
with open(os.path.join(CKPT_DIR, 'config.json'), 'w') as f:
  json.dump({'run_id': RUN_ID, 'checkpoint_modelo': os.path.basename(globals().get('best_ckpt_path', '?')),
             'dispositivo': GPU_NOME, 'cuda': torch.version.cuda, 'precisao_forecast': 'Transformer bf16 (padrão do gptcast) + VQGAN float32/TF32',
             'amostragem': AMOSTRAGEM, 'amostragem_detalhe': {k: list(v) for k, v in CFG.items()}, 'amostragem_texto': AMOSTRAGEM_TXT,
             'n_janelas': N_EVAL, 'semente_base': SEED, 'metodos': METHODS, 'chunk': CHUNK,
             'primeira_janela_t0': str(ts_t0[0]), 'ultima_janela_t0': str(ts_t0[-1])}, f, indent=2, ensure_ascii=False)


@contextlib.contextmanager
def _silencio():
  """Esconde as mensagens repetitivas do GPTCast e do LINDA (dezenas de linhas por janela)."""
  with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
    warnings.simplefilter('ignore')
    yield


def _sync() -> float:
  if torch.cuda.is_available():
    torch.cuda.synchronize()
  return time.perf_counter()


def _rain(dbz):
  return np.clip(dbz_to_rainfall(np.clip(np.asarray(dbz, dtype=np.float64), 0.0, 60.0)), 0.0, 205.0).astype(np.float32)


# Gerador do GPTCast com a amostragem exposta: reproduz o predict_sequence() do pacote (o que o forecast() chama;
# a Célula 4.5 conferiu que, com temperatura 1 e sem top-k, o resultado é idêntico ao forecast()). Contexto de 7
# quadros empilhado e codificado pela VQGAN em float32, tokens do Transformer em bf16 na janela espacial do modelo,
# cada quadro previsto decodificado isolado. Com vários sorteios, gera todos num só lote na GPU.
WS = int(math.sqrt(gpt8.transformer.config.block_size // 8))


@contextlib.contextmanager
def _lote(model, n):
  # O pacote grava cada token com predicted_indices[:, i, j] = res, e res vem com forma (lote, 1): com lote > 1 isso
  # falha. Só durante a geração em lote, o predict_next_index() deste modelo devolve (lote,); depois volta ao original.
  if n == 1:
    yield
    return
  original = model.predict_next_index
  model.predict_next_index = lambda contexto, temperature=1.0, top_k=None: original(contexto, temperature, top_k)[:, 0]
  try:
    yield
  finally:
    del model.predict_next_index


@torch.inference_mode()                       # sem registro de gradientes: menos memória e trabalho na GPU
def _gptcast(model, ctx_dbz, seed, cfg=(1.0, None, 1)):
  temperatura, top_k, n = cfg
  torch.manual_seed(seed)
  np.random.seed(seed)
  x = torch.tensor(np.clip(np.nan_to_num(ctx_dbz, nan=0.0), 0, 60) / 30.0 - 1.0, dtype=torch.float32, device=DEVICE)
  s = x.shape[0]
  xs = einops.rearrange(x, 's h w -> (s h) w')[None, None].contiguous()
  quant, idx = model.encode_to_z(xs)
  _, c_idx = model.encode_to_c(xs)
  qs = quant.shape
  ind = idx.reshape(qs[0], s, qs[2] // s, qs[3]).repeat(n, 1, 1, 1)
  c_idx = c_idx.repeat(n, 1)
  ih, iw = ind.shape[2], ind.shape[3]
  quadros = []
  for _ in range(FORECAST_STEPS):
    with torch.autocast('cuda', dtype=torch.bfloat16), _lote(model, n):
      p = model.predict_next_frame_indices(ind[:, -s:], c_idx, WS, temperature=temperatura, top_k=top_k,
                                           show_progress=False)
    quadros.append(model.decode_to_img(p.reshape(n, -1), torch.Size([n, qs[1], ih, iw])))
    ind = torch.cat((ind, p[:, None]), dim=1)
  y = torch.cat(quadros, dim=1).float().cpu().numpy().clip(-1, 1)
  r = _rain((y + 1.0) * 30.0)                 # (n, passos, H, W)
  return r.mean(axis=0) if n > 1 else r[0]


# ------------------------------------------------- 2. execução real, retomável por blocos
_pys = PyStepsNowcastAdapter('LK')
with _silencio():
  _ = _gptcast(gpt8, np.clip(seqs_dbz[0, :CONTEXT_STEPS], 0, 60), SEED, CFG['ft'])   # aquecimento da GPU
chunks = list(range(0, N_EVAL, CHUNK))

# Blocos antigos sem registro de dispositivo: os gerados na CPU (antes da correção de 17/11/2026) usam outro gerador
# aleatório e são separados para não misturar origens. O critério é o tempo do Fine-Tuned por janela: acima de 5x o
# tempo medido na GPU pela Célula 4 (ou de 20 s, sem essa medida), o bloco é considerado da CPU.
_t_gpu = (globals().get('timing_gpu_log') or {}).get('inferencia_4passos_finetuned_s')
LIMIAR_CPU_S = 5 * float(_t_gpu) if _t_gpu else 20.0
if SEPARAR_BLOCOS_CPU:
  _sep = os.path.join(CKPT_DIR, '_separados_gerados_na_cpu')
  for c in chunks:
    _p = os.path.join(CKPT_DIR, f'bloco_{c:05d}.npz')
    if not os.path.exists(_p):
      continue
    with np.load(_p) as z:
      _disp = str(z['dispositivo']) if 'dispositivo' in z.files else None
      _t_ft = float(np.median(np.abs(z['tempo_ft'])))
    if _disp is None and _t_ft > LIMIAR_CPU_S:
      os.makedirs(_sep, exist_ok=True)
      os.replace(_p, os.path.join(_sep, os.path.basename(_p)))
      print(f'[Célula 12] Bloco {c // CHUNK + 1} separado: {_t_ft:.1f} s por janela no Fine-Tuned (limiar {LIMIAR_CPU_S:.1f} s), '
            'provavelmente gerado na CPU. Será refeito na GPU.')

done = [c for c in chunks if os.path.exists(os.path.join(CKPT_DIR, f'bloco_{c:05d}.npz'))]
faltam = N_EVAL - sum(min(CHUNK, N_EVAL - c) for c in done)
print(f'[Célula 12] Amostragem do GPTCast: {AMOSTRAGEM_TXT} ({AMOSTRAGEM})')
for _d, _ks in REUSO:
  if os.path.isdir(_d):
    print(f'[Célula 12] Reaproveita {", ".join(_ks)} de {os.path.basename(_d)} nos blocos que existirem lá.')
print(f'[Célula 12] Dispositivo: {GPU_NOME} | CUDA {torch.version.cuda} | GPTCast com a precisão padrão do pacote (Transformer bf16, VQGAN float32 + TF32), modo de inferência')
print(f'[Célula 12] {N_EVAL} janelas de teste (t0 de {str(ts_t0[0])[:16]} a {str(ts_t0[-1])[:16]}) | '
      f'{len(done)} de {len(chunks)} blocos já no Drive | faltam {faltam} janelas | pasta {os.path.basename(CKPT_DIR)}')
if faltam:
  print('[Célula 12] O laço tem fim certo: para ao completar as janelas que faltam. Pode interromper quando quiser; '
        'os blocos salvos ficam no Drive e a próxima execução continua do primeiro bloco ausente.')

t_run0, n_feitas, tempos_j = time.perf_counter(), 0, []
for c in chunks:
  path = os.path.join(CKPT_DIR, f'bloco_{c:05d}.npz')
  if os.path.exists(path):
    continue
  idx = np.arange(c, min(c + CHUNK, N_EVAL))
  preds = {k: np.empty((len(idx), FORECAST_STEPS, GRID_SIZE, GRID_SIZE), np.float16) for k in KEYS.values()}
  times = {k: np.empty(len(idx), np.float64) for k in KEYS.values()}
  reuso = {}
  for _d, _ks in REUSO:
    _bp = os.path.join(_d, f'bloco_{c:05d}.npz')
    if os.path.exists(_bp):
      with np.load(_bp) as z:
        if np.array_equal(z['idx'], idx) and all(f'pred_{k}' in z.files for k in _ks):
          reuso.update({k: (z[f'pred_{k}'], z[f'tempo_{k}']) for k in _ks})
  for k, (pv, tv) in reuso.items():
    preds[k][:], times[k][:] = pv, tv
  for j, i in enumerate(idx):
    t_j = time.perf_counter()
    ctx = np.clip(np.asarray(seqs_dbz[i, :CONTEXT_STEPS], dtype=np.float32), 0.0, 60.0)
    with _silencio():
      for k, model, sem in (('ft', gpt8, 0), ('zs', gpt8_zeroshot, 0), ('ft_media', gpt8, 100_000)):
        if k in CFG and k not in reuso:
          t0 = _sync(); preds[k][j] = _gptcast(model, ctx, SEED + int(i) + sem, CFG[k]); times[k][j] = _sync() - t0
      if 'linda' not in reuso:
        ctx_rain = _rain(ctx)
        t0 = time.perf_counter()
        vel = _pys.compute_motion_field(ctx)
        t_of = time.perf_counter() - t0
        t0 = time.perf_counter(); preds['lk'][j] = _pys.forecast_extrapolation_lk(ctx_rain[-1], vel, FORECAST_STEPS); times['lk'][j] = t_of + time.perf_counter() - t0
        t0 = time.perf_counter(); preds['linda'][j] = _pys.forecast_linda(ctx_rain, vel, FORECAST_STEPS); times['linda'][j] = t_of + time.perf_counter() - t0
        if 'FALHOU' in _pys.linda_status:
          times['linda'][j] = -times['linda'][j]   # tempo negativo marca janela em que o LINDA caiu para o LK
        t0 = time.perf_counter(); preds['pers'][j] = np.repeat(ctx_rain[-1:], FORECAST_STEPS, axis=0); times['pers'][j] = time.perf_counter() - t0
    n_feitas += 1
    tempos_j.append(time.perf_counter() - t_j)
    if n_feitas == 1 or n_feitas % 10 == 0:
      media = float(np.mean(tempos_j[-50:]))
      print(f'  {N_EVAL - faltam + n_feitas}/{N_EVAL} janelas ({100 * (N_EVAL - faltam + n_feitas) / N_EVAL:.1f} %) | '
            f'{media:.1f} s/janela | faltam ~{(faltam - n_feitas) * media / 3600:.2f} h', flush=True)
  tmp = path.replace('.npz', '.tmp.npz')
  np.savez_compressed(tmp, idx=idx, dispositivo=np.array(GPU_NOME), **{f'pred_{k}': v for k, v in preds.items()}, **{f'tempo_{k}': v for k, v in times.items()})
  os.replace(tmp, path)
  feitos = len([f for f in glob.glob(os.path.join(CKPT_DIR, 'bloco_*.npz')) if not f.endswith('.tmp.npz')])
  print(f'  bloco {c // CHUNK + 1}/{len(chunks)} salvo no Drive ({feitos} de {len(chunks)} | '
        f'{(time.perf_counter() - t_run0) / 60:.1f} min nesta sessão)', flush=True)

# ------------------------------------------------------------------ 3. montagem
obs_rain_all = _rain(seqs_dbz[:, CONTEXT_STEPS:])
pred_all = {k: np.empty((N_EVAL, FORECAST_STEPS, GRID_SIZE, GRID_SIZE), np.float32) for k in KEYS.values()}
time_all = {k: np.empty(N_EVAL, np.float64) for k in KEYS.values()}
for c in chunks:
  with np.load(os.path.join(CKPT_DIR, f'bloco_{c:05d}.npz')) as z:
    for k in KEYS.values():
      pred_all[k][z['idx']] = z[f'pred_{k}'].astype(np.float32)
      time_all[k][z['idx']] = z[f'tempo_{k}']
n_linda_fail = int(np.sum(time_all['linda'] < 0))
time_all['linda'] = np.abs(time_all['linda'])

CELL12_FULL_2025_RESULTS = {
    'obs_rain': obs_rain_all, 'ft_rain': pred_all['ft'], 'zs_rain': pred_all['zs'],
    'linda_rain': pred_all['linda'], 'lk_rain': pred_all['lk'], 'pers_rain': pred_all['pers'],
    'timestamps_t0': ts_t0, 'timestamps_fim': ts_end, 'n_sequences': N_EVAL, 'run_id': RUN_ID,
    'amostragem': AMOSTRAGEM, 'amostragem_texto': AMOSTRAGEM_TXT,
}
if 'ft_media' in pred_all:
  CELL12_FULL_2025_RESULTS['ft_media_rain'] = pred_all['ft_media']


# ----------------------------------------------------------------------- 4. métricas
def _dbz(r):
  out = np.zeros_like(r, dtype=np.float64)
  w = r > 0.0365
  out[w] = rainfall_to_dbz(r[w].astype(np.float64))
  return np.clip(out, 0.0, 60.0)


def _cont(o, p, thr):
  ob, pb = o >= thr, p >= thr
  tp = float(np.sum(ob & pb)); fp = float(np.sum(~ob & pb)); fn = float(np.sum(ob & ~pb)); tn = float(ob.size) - tp - fp - fn
  return tp, fp, fn, tn


def _scores(tp, fp, fn, tn):
  n = tp + fp + fn + tn
  csi = tp / (tp + fp + fn) if tp + fp + fn else np.nan
  bias = (tp + fp) / (tp + fn) if tp + fn else np.nan
  pod = tp / (tp + fn) if tp + fn else np.nan
  far = fp / (tp + fp) if tp + fp else np.nan
  hss_den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
  hss = 2 * (tp * tn - fp * fn) / hss_den if hss_den else np.nan
  tpr = (tp + fn) * (tp + fp) / n if n else np.nan
  ets = (tp - tpr) / (tp + fp + fn - tpr) if n and (tp + fp + fn - tpr) else np.nan
  return csi, bias, pod, far, hss, ets


def metrics_block(o, p, name, lead_label, ssim_vals):
  """o, p: (N, H, W) ou (N, passos, H, W) em mm/h; ssim_vals: SSIM já calculado de cada quadro."""
  o64, p64 = o.astype(np.float64), p.astype(np.float64)
  d = p64 - o64
  n_frames = int(np.prod(o.shape[:-2]))
  row = {'Ambiente_Execucao': ENV_LABEL, 'Hardware_Tag': ENV_TAG, 'Modelo': name, 'Lead Time': lead_label,
         'Janelas': N_EVAL, 'Quadros': n_frames,
         'MAE [mm/h] (↓)': round(float(np.mean(np.abs(d))), 4),
         'RMSE [mm/h] (↓)': round(float(np.sqrt(np.mean(d ** 2))), 4),
         'SSIM dBZ (↑)': round(float(np.mean(ssim_vals)), 4),
         'PCC (↑)': round(float(np.corrcoef(o64.ravel(), p64.ravel())[0, 1]), 4),
         'Volume Prev/Obs (=1)': round(float(p64.sum() / o64.sum()), 3)}
  for thr in THRESHOLDS:
    csi, bias, pod, far, hss, ets = _scores(*_cont(o, p, thr))
    row[f'CSI @ {thr}'] = round(csi, 4)
    row[f'BIAS @ {thr}'] = round(bias, 3)
    row[f'POD @ {thr}'] = round(pod, 4)
    row[f'FAR @ {thr}'] = round(far, 4)
    row[f'HSS @ {thr}'] = round(hss, 4)
    row[f'ETS @ {thr}'] = round(ets, 4)
  return row


print('[Célula 12] Previsões completas. Etapa 1/3: métricas e SSIM por método (alguns minutos) ...', flush=True)
obs_dbz_frames = np.stack([_dbz(obs_rain_all[i]) for i in range(N_EVAL)])   # (N, passos, H, W)
rows_lead, rows_all = [], []
for name in METHODS:
  P = pred_all[KEYS[name]]
  ss = np.array([[ssim(obs_dbz_frames[i, s], _dbz(P[i, s]), data_range=60.0) for s in range(FORECAST_STEPS)]
                 for i in range(N_EVAL)])
  for s in range(FORECAST_STEPS):
    rows_lead.append(metrics_block(obs_rain_all[:, s], P[:, s], name, f'+{STEP_MIN * (s + 1)} min', ss[:, s]))
  r = metrics_block(obs_rain_all, P, name, '+30 a +120 min', ss)
  r['Tempo médio 4 passos [s]'] = round(float(time_all[KEYS[name]].mean()), 4)
  r['Tempo total [s]'] = round(float(time_all[KEYS[name]].sum()), 1)
  rows_all.append(r)
  print(f'  métricas prontas: {name}', flush=True)
df_lead = pd.DataFrame(rows_lead)
df_cell12_full = pd.DataFrame(rows_all)

# --------------------------------------- 5. bootstrap em blocos diários (janelas se sobrepõem)
print('[Célula 12] Etapa 2/3: bootstrap em blocos diários ...', flush=True)
days = pd.to_datetime(ts_t0).normalize().values
uniq_days = np.unique(days)
day_idx = [np.flatnonzero(days == d) for d in uniq_days]
thr_b = 0.5
per_win = {}
for name in METHODS:
  P = pred_all[KEYS[name]]
  ob, pb = obs_rain_all >= thr_b, P >= thr_b
  per_win[name] = {
      'tp': np.sum(ob & pb, axis=(1, 2, 3)).astype(np.float64), 'fp': np.sum(~ob & pb, axis=(1, 2, 3)).astype(np.float64),
      'fn': np.sum(ob & ~pb, axis=(1, 2, 3)).astype(np.float64),
      'ae': np.sum(np.abs(P.astype(np.float64) - obs_rain_all), axis=(1, 2, 3)),
  }
npx = FORECAST_STEPS * GRID_SIZE * GRID_SIZE
rng = np.random.default_rng(SEED)
boot = {name: {'csi': [], 'mae': []} for name in METHODS}
for _ in range(N_BOOT):
  pick = np.concatenate([day_idx[k] for k in rng.integers(0, len(day_idx), len(day_idx))])
  for name in METHODS:
    w = per_win[name]
    tp, fp, fn = w['tp'][pick].sum(), w['fp'][pick].sum(), w['fn'][pick].sum()
    boot[name]['csi'].append(tp / (tp + fp + fn) if tp + fp + fn else np.nan)
    boot[name]['mae'].append(w['ae'][pick].sum() / (len(pick) * npx))
ref = 'GPTCast-8x8 (Fine-Tuned RMF)'
rows_boot = []
for name in METHODS:
  c, m = np.array(boot[name]['csi']), np.array(boot[name]['mae'])
  r = {'Modelo': name, f'CSI ≥ {thr_b} (IC 95%)': f'{np.nanpercentile(c, 2.5):.3f} a {np.nanpercentile(c, 97.5):.3f}',
       'MAE mm/h (IC 95%)': f'{np.percentile(m, 2.5):.4f} a {np.percentile(m, 97.5):.4f}'}
  if name != ref:
    dc = np.array(boot[ref]['csi']) - c
    dm = np.array(boot[ref]['mae']) - m
    r[f'Ajustado − este: ΔCSI (IC 95%)'] = f'{np.nanpercentile(dc, 2.5):+.3f} a {np.nanpercentile(dc, 97.5):+.3f}'
    r['Ajustado − este: ΔMAE (IC 95%)'] = f'{np.percentile(dm, 2.5):+.4f} a {np.percentile(dm, 97.5):+.4f}'
    r['Leitura'] = ('ajustado melhor em CSI' if np.nanpercentile(dc, 2.5) > 0 else
                    'ajustado pior em CSI' if np.nanpercentile(dc, 97.5) < 0 else 'diferença de CSI não significativa')
  rows_boot.append(r)
df_boot = pd.DataFrame(rows_boot)

# ------------------------------------------------- 6. FSS e SAL no conjunto (Células 11 e 8)
print('[Célula 12] Etapa 3/3: FSS e SAL no conjunto (alguns minutos) ...', flush=True)
preds_named = {name: pred_all[KEYS[name]] for name in METHODS}
df_fss_2025 = fss_over_set(obs_rain_all, preds_named, thr=0.5) if 'fss_over_set' in globals() else pd.DataFrame()
sel = slice(None, None, SAL_STRIDE)
df_sal_2025 = (sal_over_set(obs_rain_all[sel], {n: p[sel] for n, p in preds_named.items()})
               if 'sal_over_set' in globals() else pd.DataFrame())

# ---------------------------------------------------------------- 7. exportação
def _save(df, nome):
  path = os.path.join(RESULTS_ENV_DIR, nome)
  df.to_csv(path, index=False)
  return path

p_all = _save(df_cell12_full, f'tabela_avaliacao_integral_1729_seq_teste_2025_{ENV_TAG}{SUF}.csv')
_save(df_lead, f'benchmark_agregado_teste2025_imerg_fr_2km_{ENV_TAG}_N{N_EVAL}{SUF}.csv')
_save(df_boot, f'bootstrap_diario_teste2025_{ENV_TAG}_N{N_EVAL}{SUF}.csv')
if len(df_fss_2025):
  _save(df_fss_2025, f'fss_conjunto_teste2025_{ENV_TAG}_N{N_EVAL}{SUF}.csv')
if len(df_sal_2025):
  _save(df_sal_2025, f'sal_conjunto_teste2025_{ENV_TAG}_N{N_EVAL}{SUF}.csv')
_save(pd.DataFrame([{'Ambiente_Execucao': ENV_LABEL, 'Hardware_Tag': ENV_TAG, 'Modelo': n, 'Amostras_Teste_2025': N_EVAL,
                     'Tempo_Medio_4Passos_s': round(float(time_all[KEYS[n]].mean()), 4),
                     'Desvio_Padrao_4Passos_s': round(float(time_all[KEYS[n]].std()), 4),
                     'Tempo_Medio_Por_Passo_s': round(float(time_all[KEYS[n]].mean() / FORECAST_STEPS), 4),
                     'Tempo_Total_Lote_s': round(float(time_all[KEYS[n]].sum()), 2),
                     'Dispositivo': f'GPU ({GPU_NOME})' if 'GPTCast' in n else 'CPU'} for n in METHODS]),
      f'velocidade_inferencia_teste2025_{ENV_TAG}_N{N_EVAL}{SUF}.csv')

print(f'\n=== Teste independente de 2025 — {N_EVAL} janelas, {N_EVAL * FORECAST_STEPS} quadros previstos, '
      f'{len(uniq_days)} dias | execução {RUN_ID} | {ENV_LABEL} ===')
print(f'Amostragem do GPTCast: {AMOSTRAGEM_TXT} (escolhida na validação de 2024, Célula 4.5)' if AMOSTRAGEM != 'padrao'
      else 'Amostragem do GPTCast: padrão do forecast() (temperatura 1, sem top-k), um sorteio por janela')
if n_linda_fail:
  print(f'[Aviso] O LINDA falhou em {n_linda_fail} janelas; nelas a previsão do LINDA é a do Lucas-Kanade.')
cols = ['Modelo', 'MAE [mm/h] (↓)', 'RMSE [mm/h] (↓)', 'SSIM dBZ (↑)', 'PCC (↑)', 'Volume Prev/Obs (=1)'] + \
       [f'CSI @ {t}' for t in THRESHOLDS] + ['HSS @ 0.5', 'ETS @ 0.5', 'BIAS @ 0.5', 'Tempo médio 4 passos [s]']
display(df_cell12_full[cols].set_index('Modelo'))
print('\n--- Por horizonte (CSI, BIAS e HSS em ≥ 0,5 mm/h) ---')
display(df_lead.pivot(index='Modelo', columns='Lead Time', values=['MAE [mm/h] (↓)', 'CSI @ 0.5', 'HSS @ 0.5', 'BIAS @ 0.5'])
        .reindex(columns=[f'+{STEP_MIN * (s + 1)} min' for s in range(FORECAST_STEPS)], level=1))
print(f'\n--- Bootstrap em blocos diários ({N_BOOT} reamostragens de {len(uniq_days)} dias): IC 95% e diferenças pareadas ---')
display(df_boot.set_index('Modelo'))
if len(df_fss_2025):
  print('\n--- FSS no conjunto (≥ 0,5 mm/h) ---')
  display(df_fss_2025.set_index(['Modelo', 'Lead Time']))
if len(df_sal_2025):
  print(f'\n--- SAL no conjunto (mediana e IQR; 1 de cada {SAL_STRIDE} janelas) ---')
  display(df_sal_2025.set_index(['Modelo', 'Lead Time']))
print(f'\n[Célula 12] Tabelas em {RESULTS_ENV_DIR} | previsões em {CKPT_DIR}')
