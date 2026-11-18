# @title
# Célula 13.1: Teste Cego Fora do Período (2026) — mesma ETL do treino, mesmos métodos e mesma decodificação
# do GPTCast da Célula 12 (escolhida na validação de 2024: F-PluvIA-G gulosa, Zero-Shot top-k 10 e, opcional,
# F-PluvIA-C8 média de 8), série da quadra chuvosa de 2026 + estudos de caso, comparada com 2025 nos mesmos meses.
#
import calendar
import contextlib
import datetime
import einops
import glob
import hashlib
import io
import json
import math
import os
import shutil
import time
import unicodedata
import warnings

import gpm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gptcast.utils.converters import dbz_to_rainfall

ENV_TAG = globals().get('ENV_TAG', 'Ambiente_Ativo')
ENV_LABEL = globals().get('ENV_LABEL', 'Hardware Ativo')
RESULTS_ENV_DIR = globals().get('RESULTS_ENV_DIR', os.path.join(SERIES_CACHE_DIR, f'resultados_{ENV_TAG}'))
DIR_2026 = os.path.join(SERIES_CACHE_DIR, 'teste_cego_2026')          # janelas e previsões (Drive)
OUT_DIR = os.path.join(RESULTS_ENV_DIR, 'teste_cego_2026')            # tabelas e figuras
os.makedirs(DIR_2026, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------- 1. configuração
MESES_2026 = [(2026, m) for m in (2, 3, 4, 5)]          # quadra chuvosa de 2026
CASOS_2026 = {                                           # estudos de caso (janela de maior chuva no período)
    'Estratiforme longo (16–17/05/2026)': ('2026-05-16 00:00', '2026-05-17 23:59'),
    'Convectivo (10/09/2026)': ('2026-09-10 00:00', '2026-09-10 23:59'),
}
INCLUIR_DIA_CORRENTE = True                              # janela mais recente do IMERG Early
PRODUTOS_ARQUIVO = [('IMERG-FR', 'RS'), ('IMERG-LR', 'NRT')]   # Final primeiro; Late se o Final ainda não saiu
PRODUTOS_DIA_CORRENTE = [('IMERG-ER', 'NRT')]
COBERTURA_MIN = 0.90                                     # fração mínima de quadros de 30 min no período
STRIDE = int(globals().get('STRIDE_JANELAS', 2))         # mesmo passo do treino (60 min)
SEED = int(globals().get('SEED', 42))
TEMPO_MAX_MIN = 120                                      # orçamento desta execução para previsões novas (min)
                                                         # as janelas são processadas em ordem aleatória fixa:
                                                         # o que estiver pronto é uma amostra uniforme da série
MAX_JANELAS = None                                       # None = todas; um número = limita a série a uma amostra
THRESHOLDS = (0.5, 1.0, 5.0, 10.0)
N_BOOT = 1000
STEP_MIN = 30

# Decodificação do GPTCast: a MESMA da Célula 12, para que 2026 e 2025 sejam comparáveis. Por padrão segue o que a
# Célula 12 usou nesta sessão; sem ela, usa a escolhida na validação de 2024 (Célula 4.5).
_R12 = globals().get('CELL12_FULL_2025_RESULTS') or {}
AMOSTRAGEM = _R12.get('amostragem', 'escolhida_2024')      # 'escolhida_2024' ou 'padrao' (forecast() do pacote)
INCLUIR_MEDIA = 'ft_media_rain' in _R12                    # F-PluvIA-C8: cerca de 3x o tempo por janela
AMOSTRAGENS = {
    'padrao': {'ft': (1.0, None, 1), 'zs': (1.0, None, 1)},
    'escolhida_2024': {'ft': (1.0, 1, 1), 'zs': (1.0, 10, 1), 'ft_media': (1.0, None, 8)},
}
CFG = {k: v for k, v in AMOSTRAGENS[AMOSTRAGEM].items() if k != 'ft_media' or INCLUIR_MEDIA}
_esc = AMOSTRAGEM == 'escolhida_2024'
NOMES = {'ft': f"GPTCast-8x8 (Fine-Tuned RMF, {'gulosa' if _esc else 'sorteio padrão'})",
         'zs': f"GPTCast-8x8 (Zero-Shot, {'top-k 10' if _esc else 'sorteio padrão'})", 'linda': 'LINDA (pysteps)',
         'lk': 'Fluxo Óptico LK (pysteps)', 'pers': 'Persistência (Baseline)'}
if 'ft_media' in CFG:
  NOMES = {'ft': NOMES['ft'], 'ft_media': 'GPTCast-8x8 (Fine-Tuned RMF, média de 8)', **{k: v for k, v in NOMES.items() if k != 'ft'}}
SUF = '' if not _esc else ('_amostragem2024_media8' if 'ft_media' in CFG else '_amostragem2024')
DECOD_TXT = ('decodificação escolhida na validação de 2024' if _esc else 'decodificação padrão do forecast()')

faltam = [n for n in ('adaptee', 'gpt8', 'gpt8_zeroshot', 'PyStepsNowcastAdapter', 'plot_rmf_comparison',
                      'CONTEXT_STEPS', 'FORECAST_STEPS', 'GRID_SIZE') if n not in globals()]
if faltam:
  raise RuntimeError(f'Rode antes as Células 2, 4, 5 e 10. Faltam: {", ".join(faltam)}')

# A versão anterior desta célula gravava como "dados reais" campos montados com a API Open-Meteo
# (modelo, não observação) e janelas de 2019–2025 rotuladas como 2026. Esses arquivos são renomeados.
antigos = [p for p in glob.glob(os.path.join(SERIES_CACHE_DIR, 'imerg_rt_2026', '*.np[yz]'))
           + glob.glob(os.path.join(SERIES_CACHE_DIR, 'rmf_imerg_rt_2026_2km_operational_cases.npz'))
           if '_INVALIDO' not in p]
for p in antigos:
  base, ext = os.path.splitext(p)
  os.rename(p, f'{base}_INVALIDO_versao_anterior{ext}')
if antigos:
  print(f'[Célula 13.1] {len(antigos)} arquivo(s) da versão anterior renomeado(s) com o sufixo _INVALIDO.')


# ------------------------------------------------- 2. ETL de 2026 (a mesma do treino)
def _rain(dbz):
  return np.clip(dbz_to_rainfall(np.clip(np.asarray(dbz, dtype=np.float64), 0.0, 60.0)), 0.0, 205.0).astype(np.float32)


GRADE = {'lat0': round(float(adaptee.target_lats[0]), 4), 'lon0': round(float(adaptee.target_lons[0]), 4),
         'n': int(adaptee.grid_size), 'res_km': float(adaptee.res_km), 'seq_len': int(adaptee.seq_len),
         'wet_mean': adaptee.wet_mean_thr, 'wet_max': adaptee.wet_max_thr}


@contextlib.contextmanager
def _silencio():
  """Esconde as mensagens repetitivas de bibliotecas (stdout e avisos); erros continuam aparecendo."""
  buf = io.StringIO()
  with warnings.catch_warnings(), contextlib.redirect_stdout(buf):
    warnings.simplefilter('ignore')
    yield buf


def _limpar_brutos():
  shutil.rmtree(adaptee.raw_temp_dir, ignore_errors=True)
  os.makedirs(adaptee.raw_temp_dir, exist_ok=True)


_DISPONIVEL = {}


def produto_disponivel(prod, ptype, inicio, fim) -> bool:
  """Testa um único dia (o do meio do período) antes de baixar o período inteiro."""
  meio = inicio + (fim - inicio) / 2
  dia0 = datetime.datetime(meio.year, meio.month, meio.day)
  chave = (prod, ptype, dia0.strftime('%Y-%m'))
  if chave not in _DISPONIVEL:
    try:
      with _silencio():
        gpm.download(product=prod, product_type=ptype, version=7, start_time=dia0,
                     end_time=dia0 + datetime.timedelta(hours=2), progress_bar=False, verbose=False)
        ds = gpm.open_dataset(product=prod, product_type=ptype, version=7, start_time=dia0,
                              end_time=dia0 + datetime.timedelta(hours=2), variables=['precipitation'])
        ds.close()
      _DISPONIVEL[chave] = True
    except Exception:
      _DISPONIVEL[chave] = False
    finally:
      _limpar_brutos()
  return _DISPONIVEL[chave]


def janelas_periodo(tag, inicio, fim, stride, produtos, cobertura_min):
  """Baixa o período na NASA PPS, recorta, interpola a 2 km e extrai as janelas de 11 quadros com o
  mesmo critério de chuva do treino (métodos do adaptee da Célula 2). Usa o primeiro produto disponível.
  Devolve (janelas em dBZ, horário do último quadro, metadados) ou None. Nunca preenche com outro dado."""
  cache = os.path.join(DIR_2026, f'{tag}_s{stride}.npz')
  if os.path.exists(cache):
    with np.load(cache) as z:
      meta = json.loads(str(z['meta']))
      if meta.get('grade') == GRADE:
        return z['seqs_dbz'].astype(np.float32), z['ts_fim'].astype('datetime64[m]'), meta
  adaptee._ensure_gpm_configured()
  esperados = int((fim - inicio).total_seconds() // (STEP_MIN * 60)) + 1
  for prod, ptype in produtos:
    if not produto_disponivel(prod, ptype, inicio, fim):
      print(f'  {tag}: {prod} ainda não publicado na NASA PPS para este período.')
      continue
    t0 = time.perf_counter()
    try:
      with _silencio():
        gpm.download(product=prod, product_type=ptype, version=7, start_time=inicio, end_time=fim,
                     force_download=False, n_threads=4, verbose=False)
        ds = gpm.open_dataset(product=prod, product_type=ptype, version=7, start_time=inicio, end_time=fim,
                              variables=['precipitation'])
        try:
          crop = ds.gpm.crop(extent=adaptee.raw_extent).load()
        finally:
          ds.close()
    except Exception as e:
      print(f'  {tag}: {prod} ({ptype}) falhou — {type(e).__name__}: {str(e)[:150]}')
      continue
    finally:
      _limpar_brutos()
    n = int(crop.sizes.get('time', 0))
    if n < max(cobertura_min * esperados, adaptee.seq_len):
      print(f'  {tag}: {prod} com {n} de {esperados} quadros (abaixo do mínimo); tentando o próximo produto.')
      continue
    da2, metodo, nan_frac = adaptee._interpolate_to_high_res_grid(crop)
    seqs, ts = adaptee._extract_valid_sequences(da2, stride=stride)
    meta = {'produto': prod, 'tipo': ptype, 'versao': 7, 'quadros_obtidos': n, 'quadros_esperados': esperados,
            'fracao_nan': round(nan_frac, 6), 'interp': metodo, 'stride': stride, 'grade': GRADE,
            'tempo_s': round(time.perf_counter() - t0, 1),
            'processado_utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M')}
    np.savez_compressed(cache, seqs_dbz=seqs.astype(np.float16), ts_fim=ts.astype('datetime64[m]'),
                        meta=np.array(json.dumps(meta)))
    print(f'  {tag}: {prod} v7 — {n}/{esperados} quadros, {len(seqs)} janelas com chuva '
          f'({time.perf_counter() - t0:.0f} s)')
    return seqs.astype(np.float32), ts.astype('datetime64[m]'), meta
  return None


# --------------------------------------------- 3. previsões dos 5 métodos (retomáveis)
def _digest(model) -> str:
  h = hashlib.sha256()
  for k, v in sorted(model.state_dict().items()):
    if k.startswith('transformer.'):
      h.update(k.encode())
      h.update(v.detach().float().cpu().numpy().tobytes())
  return h.hexdigest()[:12]


def _sync() -> float:
  if torch.cuda.is_available():
    torch.cuda.synchronize()
  return time.perf_counter()


# Mesmo gerador da Célula 12 (reproduz o predict_sequence() do pacote, conferido como idêntico ao forecast() na
# Célula 4.5), com temperatura e top-k expostos e vários sorteios num só lote na GPU.
WS = int(math.sqrt(gpt8.transformer.config.block_size // 8))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


@contextlib.contextmanager
def _lote(model, n):
  if n == 1:
    yield
    return
  original = model.predict_next_index
  model.predict_next_index = lambda contexto, temperature=1.0, top_k=None: original(contexto, temperature, top_k)[:, 0]
  try:
    yield
  finally:
    del model.predict_next_index


@torch.inference_mode()
def _gptcast(model, ctx_dbz, seed, cfg=(1.0, None, 1)):
  temperatura, top_k, n = cfg
  torch.manual_seed(seed)
  np.random.seed(seed)
  x = torch.tensor(np.clip(np.nan_to_num(ctx_dbz, nan=0.0), 0, 60) / 30.0 - 1.0, dtype=torch.float32, device=DEV)
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
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=DEV == 'cuda'), _lote(model, n):
      p = model.predict_next_frame_indices(ind[:, -s:], c_idx, WS, temperature=temperatura, top_k=top_k,
                                           show_progress=False)
    quadros.append(model.decode_to_img(p.reshape(n, -1), torch.Size([n, qs[1], ih, iw])))
    ind = torch.cat((ind, p[:, None]), dim=1)
  y = torch.cat(quadros, dim=1).float().cpu().numpy().clip(-1, 1)
  r = _rain((y + 1.0) * 30.0)
  return r.mean(axis=0) if n > 1 else r[0]


DIGEST = _digest(gpt8)
_pys = PyStepsNowcastAdapter('LK')


# Uma pasta por decodificação (as previsões do GPTCast mudam com ela). LINDA, LK e persistência não dependem da
# decodificação e vêm da pasta padrão quando já existem lá; com a média, o guloso e o top-k 10 vêm da pasta sem média.
_BASE = os.path.join(DIR_2026, 'previsoes', f'janelas_{DIGEST}_s{SEED}')
_PASTA_A = _BASE + '_amostragem2024'
PASTA_PREV = _BASE if not _esc else (_PASTA_A + '_media8' if 'ft_media' in CFG else _PASTA_A)
REUSO = [] if not _esc else [(_BASE, ('linda', 'lk', 'pers'))] + ([(_PASTA_A, ('ft', 'zs'))] if 'ft_media' in CFG else [])
os.makedirs(PASTA_PREV, exist_ok=True)


def _arq(prod, ts_fim, pasta=None):
  return os.path.join(pasta or PASTA_PREV, f"{prod}_{np.datetime_as_string(ts_fim, unit='m').replace('-', '').replace(':', '')}.npz")


def _reaproveitar(prod, ts_fim):
  """Previsões e tempos já calculados em outras pastas para esta janela (métodos que não mudam aqui)."""
  out = {}
  for pasta, ks in REUSO:
    f = _arq(prod, ts_fim, pasta)
    if os.path.exists(f):
      with np.load(f) as z:
        if all(f'pred_{k}' in z.files for k in ks):
          out.update({k: (z[f'pred_{k}'].astype(np.float32), float(z[f'tempo_{k}']), bool(z['linda_ok'])) for k in ks})
  return out


def _uma_janela(seq_dbz, seed, reuso=None):
  """Os métodos numa janela, cronometrados, sem as mensagens do GPTCast e do LINDA; reaproveita o que já existe."""
  P, T, reuso = {}, {}, reuso or {}
  linda_ok = True
  for k, (pv, tv, ok) in reuso.items():
    P[k], T[k] = pv, tv
    if k == 'linda':
      linda_ok = ok
  ctx = np.clip(seq_dbz[:CONTEXT_STEPS].astype(np.float32), 0.0, 60.0)
  with _silencio():
    for k, model, sem in (('ft', gpt8, 0), ('zs', gpt8_zeroshot, 0), ('ft_media', gpt8, 100_000)):
      if k in CFG and k not in reuso:
        t = _sync(); P[k] = _gptcast(model, ctx, seed + sem, CFG[k]); T[k] = _sync() - t
    if 'linda' not in reuso:
      ctx_r = _rain(ctx)
      t = time.perf_counter(); vel = _pys.compute_motion_field(ctx); t_of = time.perf_counter() - t
      t = time.perf_counter(); P['lk'] = _pys.forecast_extrapolation_lk(ctx_r[-1], vel, FORECAST_STEPS); T['lk'] = t_of + time.perf_counter() - t
      t = time.perf_counter(); P['linda'] = _pys.forecast_linda(ctx_r, vel, FORECAST_STEPS); T['linda'] = t_of + time.perf_counter() - t
      t = time.perf_counter(); P['pers'] = np.repeat(ctx_r[-1:], FORECAST_STEPS, axis=0); T['pers'] = time.perf_counter() - t
      linda_ok = not str(getattr(_pys, 'linda_status', '')).startswith('FALHOU')
  return P, T, linda_ok


def prever(seqs_dbz, ts_fim, prods, seeds, rotulo, tempo_max_min=None):
  """Roda os 5 métodos nas janelas que ainda não têm previsão no Drive (um arquivo por janela), em ordem
  aleatória fixa, até acabar ou até o orçamento de tempo. Devolve previsões e máscara das prontas."""
  n = len(seqs_dbz)
  feitos = np.array([os.path.exists(_arq(prods[i], ts_fim[i])) for i in range(n)])
  ordem = [i for i in np.random.default_rng(SEED).permutation(n) if not feitos[i]]
  if ordem:
    print(f'  {rotulo}: {int(feitos.sum())}/{n} janelas já no Drive; {len(ordem)} a calcular'
          + (f' (limite desta execução: {tempo_max_min} min)' if tempo_max_min else '') + '.')
  t_ini, tempos, falhas_linda = time.perf_counter(), [], 0
  for k, i in enumerate(ordem, 1):
    t_j = time.perf_counter()
    P, T, ok = _uma_janela(seqs_dbz[i], int(seeds[i]), _reaproveitar(prods[i], ts_fim[i]))
    falhas_linda += (not ok)
    tmp = _arq(prods[i], ts_fim[i]).replace('.npz', '.tmp.npz')
    np.savez_compressed(tmp, seed=int(seeds[i]), linda_ok=ok,
                        **{f'pred_{m}': v.astype(np.float16) for m, v in P.items()},
                        **{f'tempo_{m}': v for m, v in T.items()})
    os.replace(tmp, _arq(prods[i], ts_fim[i]))
    feitos[i] = True
    tempos.append(time.perf_counter() - t_j)
    gasto = (time.perf_counter() - t_ini) / 60
    if k == 1 or k % 25 == 0 or k == len(ordem):
      resto = (len(ordem) - k) * np.mean(tempos[-50:]) / 3600
      print(f'  {rotulo}: {int(feitos.sum())}/{n} prontas ({100 * feitos.mean():.1f} %) | '
            f'{np.mean(tempos[-50:]):.1f} s/janela | faltam ~{resto:.1f} h', flush=True)
    if tempo_max_min and gasto >= tempo_max_min and k < len(ordem):
      print(f'  {rotulo}: limite de {tempo_max_min} min atingido. Rode a célula de novo para continuar; '
            f'as métricas abaixo usam as {int(feitos.sum())} janelas prontas (amostra aleatória da série).')
      break
  if falhas_linda:
    print(f'  [aviso] LINDA falhou em {falhas_linda} janela(s) e foi trocado pelo Lucas-Kanade nelas.')
  idx = np.flatnonzero(feitos)
  pred = {m: np.empty((len(idx), FORECAST_STEPS, GRID_SIZE, GRID_SIZE), np.float32) for m in NOMES}
  tempo = {m: np.empty(len(idx)) for m in NOMES}
  for j, i in enumerate(idx):
    with np.load(_arq(prods[i], ts_fim[i])) as z:
      for m in NOMES:
        pred[m][j] = z[f'pred_{m}'].astype(np.float32)
        tempo[m][j] = float(z[f'tempo_{m}'])
  return pred, tempo, feitos


def migrar_blocos_antigos(seqs_dbz, ts_fim, prods):
  """Aproveita as previsões da execução anterior (blocos de 50 janelas, mesma ordem da série completa)."""
  n, novos = len(seqs_dbz), 0
  if _esc:          # os blocos antigos foram gerados com o sorteio padrão: só servem à pasta padrão
    return
  for pasta in glob.glob(os.path.join(DIR_2026, 'previsoes', f'serie_*_{DIGEST}_n{n}_s{SEED}')):
    for f in sorted(glob.glob(os.path.join(pasta, 'bloco_*.npz'))):
      if f.endswith('.tmp.npz'):
        continue
      with np.load(f) as z:
        for j, i in enumerate(z['idx']):
          dst = _arq(prods[i], ts_fim[i])
          if not os.path.exists(dst):
            np.savez_compressed(dst, seed=SEED + int(i), linda_ok=True,
                                **{f'pred_{m}': z[f'pred_{m}'][j] for m in NOMES},
                                **{f'tempo_{m}': z[f'tempo_{m}'][j] for m in NOMES})
            novos += 1
  if novos:
    print(f'[Célula 13.1] {novos} previsões da execução anterior aproveitadas.')


# ---------------------------------------------------------------------- 4. métricas
def metricas(o, p):
  """o, p: (N, passos, H, W) em mm/h. Uma linha por horizonte e uma com todos."""
  rows = []
  for l in list(range(o.shape[1])) + [None]:
    oo = (o if l is None else o[:, l]).astype(np.float64)
    pp = (p if l is None else p[:, l]).astype(np.float64)
    d = pp - oo
    r = {'Lead': 'todos' if l is None else f'+{STEP_MIN * (l + 1)} min',
         'MAE [mm/h] (↓)': float(np.mean(np.abs(d))), 'RMSE [mm/h] (↓)': float(np.sqrt(np.mean(d ** 2))),
         'Volume Prev/Obs (=1)': float(pp.sum() / oo.sum()) if oo.sum() > 0 else np.nan}
    so, sp = oo.std(), pp.std()
    r['Correlação (↑)'] = float(np.mean((oo - oo.mean()) * (pp - pp.mean())) / (so * sp)) if so > 0 and sp > 0 else np.nan
    for thr in THRESHOLDS:
      ob, pb = oo >= thr, pp >= thr
      tp, fp, fn = float(np.sum(ob & pb)), float(np.sum(~ob & pb)), float(np.sum(ob & ~pb))
      r[f'CSI_{thr}'] = tp / (tp + fp + fn) if tp + fp + fn else np.nan
      r[f'BIAS_{thr}'] = (tp + fp) / (tp + fn) if tp + fn else np.nan
      if thr == 1.0:
        r['POD_1.0 (recall)'] = tp / (tp + fn) if tp + fn else np.nan
        r['Precisão_1.0'] = tp / (tp + fp) if tp + fp else np.nan
        r['F1_1.0'] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
    rows.append(r)
  return rows


def bootstrap_dias(o, preds, ts_t0, thr=1.0):
  """IC de 95 % por reamostragem de dias inteiros (as janelas do mesmo dia são correlacionadas)."""
  dias = pd.to_datetime(ts_t0).normalize().values
  ud, inv = np.unique(dias, return_inverse=True)
  por = {}
  for k, p in preds.items():
    ae = np.abs(p - o).reshape(len(o), -1).sum(1)
    ob, pb = o >= thr, p >= thr
    por[k] = np.stack([np.bincount(inv, ae, len(ud)),
                       np.bincount(inv, (ob & pb).reshape(len(o), -1).sum(1), len(ud)),
                       np.bincount(inv, (~ob & pb).reshape(len(o), -1).sum(1), len(ud)),
                       np.bincount(inv, (ob & ~pb).reshape(len(o), -1).sum(1), len(ud))])
  npx = np.bincount(inv, minlength=len(ud)) * o[0].size
  rng = np.random.default_rng(SEED)
  amostras = rng.integers(0, len(ud), (N_BOOT, len(ud)))
  out, est = [], {}
  for k, a in por.items():
    s = a[:, amostras].sum(-1)
    mae = s[0] / npx[amostras].sum(-1)
    csi = s[1] / np.maximum(s[1] + s[2] + s[3], 1)
    est[k] = (mae, csi)
  for k in por:
    mae, csi = est[k]
    linha = {'Método': NOMES[k], 'Dias': len(ud),
             'MAE IC95 [mm/h]': f'{np.percentile(mae, 2.5):.3f} – {np.percentile(mae, 97.5):.3f}',
             f'CSI ≥ {thr} IC95': f'{np.percentile(csi, 2.5):.3f} – {np.percentile(csi, 97.5):.3f}'}
    if k != 'ft':
      dcsi = est['ft'][1] - csi
      dmae = est['ft'][0] - mae
      linha[f'ΔCSI ≥ {thr} (FT − método) IC95'] = f'{np.percentile(dcsi, 2.5):+.3f} – {np.percentile(dcsi, 97.5):+.3f}'
      linha['ΔMAE (FT − método) IC95'] = f'{np.percentile(dmae, 2.5):+.3f} – {np.percentile(dmae, 97.5):+.3f}'
      linha['FT melhor no CSI?'] = 'sim' if np.percentile(dcsi, 2.5) > 0 else ('não' if np.percentile(dcsi, 97.5) < 0 else 'inconclusivo')
    out.append(linha)
  return pd.DataFrame(out)


# --------------------------------------------- 5. série da quadra chuvosa de 2026
print(f'[Célula 13.1] Decodificação do GPTCast: {DECOD_TXT} | ' + ', '.join(f'{NOMES[k]}' for k in CFG)
      + f' | previsões em {os.path.basename(PASTA_PREV)}')
print(f'[Célula 13.1] Série de 2026: {", ".join(f"{m:02d}/{y}" for y, m in MESES_2026)} | passo {STRIDE * STEP_MIN} min')
partes, info = [], []
for y, m in MESES_2026:
  ini = datetime.datetime(y, m, 1)
  fim = datetime.datetime(y, m, calendar.monthrange(y, m)[1], 23, 59)
  r = janelas_periodo(f'RMF_{y}_{m:02d}', ini, fim, STRIDE, PRODUTOS_ARQUIVO, COBERTURA_MIN)
  if r is None:
    info.append({'Mês': f'{m:02d}/{y}', 'Produto': 'sem dados disponíveis', 'Quadros': 0, 'Janelas': 0})
    continue
  s, t, meta = r
  partes.append((s, t, meta['produto']))
  info.append({'Mês': f'{m:02d}/{y}', 'Produto': f"{meta['produto']} v7 ({meta['tipo']})",
               'Quadros': f"{meta['quadros_obtidos']}/{meta['quadros_esperados']}", 'Janelas': len(s)})
df_meses = pd.DataFrame(info)
display(df_meses.set_index('Mês'))
if not partes:
  raise RuntimeError('Nenhum mês de 2026 disponível na NASA PPS para os produtos configurados.')

seqs26 = np.concatenate([p[0] for p in partes])
ts26_fim = np.concatenate([p[1] for p in partes])
prod26 = np.concatenate([[p[2]] * len(p[0]) for p in partes])
seeds26 = SEED + np.arange(len(seqs26))                  # semente fixa por janela da série completa
migrar_blocos_antigos(seqs26, ts26_fim, prod26)
if MAX_JANELAS and len(seqs26) > MAX_JANELAS:
  sel = np.unique(np.linspace(0, len(seqs26) - 1, MAX_JANELAS).astype(int))
  seqs26, ts26_fim, prod26, seeds26 = seqs26[sel], ts26_fim[sel], prod26[sel], seeds26[sel]
print(f'[Célula 13.1] {len(seqs26)} janelas de 2026 na série.')

pred26, tempo26, prontas = prever(seqs26, ts26_fim, prod26, seeds26, 'série 2026', TEMPO_MAX_MIN)
seqs26, ts26_fim, prod26 = seqs26[prontas], ts26_fim[prontas], prod26[prontas]
ts26_t0 = ts26_fim - np.timedelta64(STEP_MIN * FORECAST_STEPS, 'm')
N26, N26_TOTAL = int(prontas.sum()), len(prontas)
if N26 == 0:
  raise RuntimeError('Nenhuma janela de 2026 com previsão pronta. Aumente TEMPO_MAX_MIN e rode de novo.')
obs26 = _rain(seqs26[:, CONTEXT_STEPS:])
PARCIAL = N26 < N26_TOTAL
if PARCIAL:
  print(f'[Célula 13.1] Resultados PARCIAIS: {N26} de {N26_TOTAL} janelas ({100 * N26 / N26_TOTAL:.0f} %), '
        'sorteadas ao acaso na série. Rode a célula de novo para completar.')
if set(prod26) != {'IMERG-FR'}:
  print(f'[Célula 13.1] Atenção: 2026 usa {", ".join(sorted(set(prod26)))}, e o modelo foi treinado com o IMERG-FR '
        '(Final, calibrado com pluviômetros). A comparação com 2025 mistura o efeito do ano com o da troca de produto.')

rows = []
for k, nome in NOMES.items():
  for r in metricas(obs26, pred26[k]):
    rows.append({'Método': nome, 'Período': '2026', **r})
df26 = pd.DataFrame(rows)

# Mesmos meses do teste de 2025 (Célula 12), para separar o efeito do ano do efeito da estação.
meses_alvo = {m for _, m in MESES_2026}
if 'CELL12_FULL_2025_RESULTS' in globals():
  R12 = CELL12_FULL_2025_RESULTS
  ts25 = pd.to_datetime(np.asarray(R12['timestamps_t0']))
  m25 = np.isin(ts25.month, list(meses_alvo))
  if R12.get('amostragem', 'padrao') != AMOSTRAGEM:
    print(f"[Aviso Célula 13.1] A Célula 12 desta sessão usou a decodificação '{R12.get('amostragem', 'padrao')}' e esta "
          f"célula usa '{AMOSTRAGEM}': a comparação com 2025 mistura decodificações diferentes.")
  for k, nome in NOMES.items():
    if f'{k}_rain' not in R12:
      continue
    for r in metricas(R12['obs_rain'][m25], R12[f'{k}_rain'][m25]):
      rows.append({'Método': nome, 'Período': f'2025 (mesmos meses, {int(m25.sum())} janelas)', **r})
  df26 = pd.DataFrame(rows)
else:
  print('[Célula 13.1] CELL12_FULL_2025_RESULTS ausente: sem comparação com 2025 (rode a Célula 12).')

df_boot = bootstrap_dias(obs26, pred26, ts26_t0)

# ------------------------------------------------------------------- 6. estudos de caso
casos = {}
for nome_caso, (a, b) in CASOS_2026.items():
  ini, fim = pd.Timestamp(a).to_pydatetime(), pd.Timestamp(b).to_pydatetime()
  tag = 'caso_' + ini.strftime('%Y%m%d') + '_' + fim.strftime('%Y%m%d')
  r = janelas_periodo(tag, ini, fim, 1, PRODUTOS_ARQUIVO, COBERTURA_MIN)
  if r is None or len(r[0]) == 0:
    print(f'[Célula 13.1] {nome_caso}: sem dados ou sem janela com chuva pelo critério do treino; caso omitido.')
    continue
  s, t, meta = r
  i = int(np.argmax(_rain(s).sum(axis=(1, 2, 3))))      # janela de maior chuva observada (11 quadros)
  casos[nome_caso] = (s[i:i + 1], t[i], meta['produto'], 'janela de maior chuva no período')

if INCLUIR_DIA_CORRENTE:
  agora = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None, second=0, microsecond=0)
  agora = agora.replace(minute=30 if agora.minute >= 30 else 0)
  ini = agora - datetime.timedelta(hours=30)
  r = janelas_periodo('dia_corrente_' + agora.strftime('%Y%m%dT%H%M'), ini, agora, 1, PRODUTOS_DIA_CORRENTE, 0.0)
  if r is None or len(r[0]) == 0:
    print('[Célula 13.1] Dia corrente: IMERG Early sem janela com chuva nas últimas 30 h (ou indisponível); omitido.')
  else:
    s, t, meta = r
    agora_brt = agora - datetime.timedelta(hours=3)
    casos[f'Dia corrente ({agora_brt:%d/%m/%Y}, horário de Brasília)'] = (s[-1:], t[-1], meta['produto'], 'janela mais recente disponível')

rows_casos = []
for nome_caso, (s, t_fim, prod, criterio) in casos.items():
  p, _, _ = prever(s, np.array([t_fim]), np.array([prod]), np.array([SEED]), nome_caso)
  o = _rain(s[:, CONTEXT_STEPS:])
  for k, nome in NOMES.items():
    for r in metricas(o, p[k]):
      if r['Lead'] != 'todos':
        rows_casos.append({'Caso': nome_caso, 'Produto': prod, 'Seleção': criterio, 'Método': nome, **r})
  t0_txt = np.datetime_as_string(t_fim - np.timedelta64(STEP_MIN * FORECAST_STEPS, 'm'), unit='m').replace('T', ' ')
  plot_rmf_comparison({f'Observado ({prod})': o[0], **{NOMES[k]: p[k][0] for k in NOMES}},
                      _rain(s[0, CONTEXT_STEPS - 1]), t_fim,
                      f'Teste cego 2026 — {nome_caso} | t0 = {t0_txt} UTC | {criterio}',
                      save_filename=os.path.join('teste_cego_2026', 'caso_' + ''.join(c for c in unicodedata.normalize('NFKD', nome_caso) if c.isascii() and c.isalnum())[:40] + f'_{ENV_TAG}{SUF}.png'))
df_casos = pd.DataFrame(rows_casos)

# ----------------------------------------------------------------- 7. saídas
def _salvar(df, nome):
  path = os.path.join(OUT_DIR, f'{nome}_{ENV_TAG}{SUF}.csv')
  df.to_csv(path, index=False)
  return path


_salvar(df_meses, 'teste_cego_2026_meses_e_produtos')
p_met = _salvar(df26, 'teste_cego_2026_metricas_por_lead')
_salvar(df_boot, 'teste_cego_2026_bootstrap_dias')
if len(df_casos):
  _salvar(df_casos, 'teste_cego_2026_estudos_de_caso')

cols = ['MAE [mm/h] (↓)', 'CSI_1.0', 'BIAS_1.0', 'CSI_5.0', 'F1_1.0', 'Volume Prev/Obs (=1)']
print(f'\n=== Teste cego 2026 ({N26} de {N26_TOTAL} janelas{" — PARCIAL" if PARCIAL else ""}, '
      f'{", ".join(sorted(set(prod26)))}) x teste 2025 (IMERG-FR) nos mesmos meses — todos os horizontes | {DECOD_TXT} | {ENV_LABEL} ===')
display(df26[df26['Lead'] == 'todos'].pivot_table(index='Método', columns='Período', values=cols, sort=False).round(3))
print('\n=== 2026 por horizonte (CSI ≥ 1 mm/h) ===')
display(df26[(df26['Período'] == '2026') & (df26['Lead'] != 'todos')]
        .pivot_table(index='Método', columns='Lead', values='CSI_1.0', sort=False).round(3))
print(f'\n=== Intervalos de confiança de 95 % (bootstrap por dias, {N_BOOT} reamostragens) ===')
display(df_boot.set_index('Método'))
if len(df_casos):
  print('\n=== Estudos de caso (uma janela cada; ' + ('F-PluvIA-G determinístico, Zero-Shot uma amostra top-k 10' if _esc
        else 'GPTCast é uma única amostra estocástica') + ') ===')
  display(df_casos.pivot_table(index=['Caso', 'Método'], columns='Lead', values='CSI_1.0', sort=False).round(3))

fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), constrained_layout=True)
cores = {'ft': '#d62728', 'ft_media': '#9467bd', 'zs': '#ff7f0e', 'linda': '#1f77b4', 'lk': '#2ca02c', 'pers': '#7f7f7f'}
x = [STEP_MIN * (i + 1) for i in range(FORECAST_STEPS)]
for k, nome in NOMES.items():
  for per, ls in (('2026', '-'), (None, '--')):
    g = df26[(df26['Método'] == nome) & (df26['Lead'] != 'todos')]
    g = g[g['Período'] == '2026'] if per else g[g['Período'] != '2026']
    if len(g):
      axes[0].plot(x, g['CSI_1.0'], ls, marker='o', color=cores[k], label=f'{nome} — {per or "2025"}')
      axes[1].plot(x, g['MAE [mm/h] (↓)'], ls, marker='o', color=cores[k])
for ax, t in zip(axes, ['CSI ≥ 1 mm/h (↑)', 'MAE (mm/h, ↓)']):
  ax.set_title(t, fontweight='bold')
  ax.set_xlabel('Horizonte (min)')
  ax.set_xticks(x)
  ax.grid(True, ls=':', alpha=0.6)
axes[0].legend(fontsize=7, ncol=2)
fig.suptitle(f'Teste cego 2026{" (parcial: " + str(N26) + " de " + str(N26_TOTAL) + " janelas)" if PARCIAL else ""} (linha cheia) x teste 2025 nos mesmos meses (tracejada) | {DECOD_TXT} | {ENV_LABEL}',
             fontsize=12.5, fontweight='bold')
png = os.path.join(OUT_DIR, f'painel_teste_cego_2026_vs_2025_{ENV_TAG}{SUF}.png')
fig.savefig(png, dpi=300, bbox_inches='tight')
plt.show()

CELL13_1_RESULTS = {'parcial': PARCIAL, 'janelas_prontas': N26, 'janelas_total': N26_TOTAL, 'obs_rain': obs26, **{f'{k}_rain': v for k, v in pred26.items()},
                    'timestamps_t0': ts26_t0, 'produto': prod26, 'metricas': df26, 'bootstrap': df_boot,
                    'casos': df_casos, 'amostragem': AMOSTRAGEM}
print(f'[Célula 13.1] Tabelas e figuras -> {OUT_DIR}')
