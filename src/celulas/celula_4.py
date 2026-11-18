# @title
# Célula 4: Montagem da Série (2019–2025), Fine-Tuning do GPTCast-8x8 na RMF (BF16), Registro
# Honesto da Origem do Modelo e Inferência Cronometrada do Estudo de Caso (18/03/2024).
#
#
import copy
import datetime
import glob
import json
import os
import re
import shutil
import time

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl_modern
import pytorch_lightning as pl_legacy
from gptcast.models import GPTCast
from gptcast.utils.converters import dbz_to_rainfall

# ------------------------------------------------------------------ 1. parâmetros
CONTEXT_STEPS = 7          # 3,5 h de contexto
FORECAST_STEPS = 4         # +30, +60, +90, +120 min
TOTAL_WINDOW = CONTEXT_STEPS + FORECAST_STEPS
RES_KM = 2.0
GRID_SIZE = 128
STRIDE_JANELAS = 2         # passo entre janelas = 2 x 30 min = 60 min (deve casar com o cache da Célula 2)
SEED = 42
CASE_STUDY_ISO = '2024-03-18T10:30'   # evento da partição de VALIDAÇÃO (2024)
FORCE_RETRAIN = False                 # True = refaz o fine-tuning mesmo havendo checkpoint válido
VAL_CROP = 'end'                      # validação com os quadros 4 a 11 (inclui os 4 a prever); 'start' = quadros 1 a 8
USAR_CHECKPOINT_QUADRO_A_QUADRO = False  # True = carrega o checkpoint antigo, treinado com tokenização quadro a
                                         # quadro (incompatível com a previsão); só para reproduzir números antigos
TOKENIZACAO_TREINO = '[OCULTO]'      # quadros empilhados na vertical, como no GPTCast original e no forecast()
N_TIMING_REPEATS = 5                  # repetições cronometradas da inferência (após 1 aquecimento)

ENV_TAG = globals().get('ENV_TAG', 'Ambiente_Ativo')
ENV_LABEL = globals().get('ENV_LABEL', 'Hardware Ativo')
CHECKPOINTS_ENV_DIR = globals().get('CHECKPOINTS_ENV_DIR', os.path.join(CHECKPOINTS_DIR, ENV_TAG))
RESULTS_ENV_DIR = globals().get('RESULTS_ENV_DIR', os.path.join(SERIES_CACHE_DIR, f'resultados_{ENV_TAG}'))
os.makedirs(CHECKPOINTS_ENV_DIR, exist_ok=True)
os.makedirs(RESULTS_ENV_DIR, exist_ok=True)

# Checkpoints antigos na pasta raiz são copiados (uma vez) para a pasta deste ambiente.
for ckpt_f in glob.glob(os.path.join(CHECKPOINTS_DIR, 'gptcast8_rmf_2km*.ckpt')):
  dst = os.path.join(CHECKPOINTS_ENV_DIR, os.path.basename(ckpt_f))
  if not os.path.exists(dst):
    shutil.copy2(ckpt_f, dst)

# ------------------------------------------------------- 2. série (Célula 2 -> Célula 3)
#
# Credenciais NASA PPS por Segurança para evitar vazamento de informaçõ de Login e Senha, ela foram alocadas em variáveis privadas
# Secret Keys fora do Ambiente de Execução desse Jupyter Notebook em questão ['PPS_USERNAME'] = 'e-mail cadastrado na PPS'..
#
try:
  from google.colab import userdata  # Segredos do Colab, se configurados
  os.environ.setdefault('PPS_USERNAME', userdata.get('PPS_USERNAME') or '')
except Exception:
  pass

adaptee = IMERGExtractorAdaptee(
    username_pps=os.environ.get('PPS_USERNAME') or None,
    password_pps=os.environ.get('PPS_PASSWORD') or None,
    raw_temp_dir=GPM_RAW_TEMP_DIR, cache_dir=SERIES_CACHE_DIR, results_env_dir=RESULTS_ENV_DIR,
    env_tag=ENV_TAG, env_label=ENV_LABEL, product='IMERG-FR', version=7,
    grid_size=GRID_SIZE, res_km=RES_KM, seq_len=TOTAL_WINDOW,
)

_ram_cache_ready = (
    'md_rmf' in globals() and getattr(md_rmf, 'all_sequences_dbz', None) is not None
    and len(md_rmf.all_sequences_dbz) > 0
)
t_etl0 = time.perf_counter()
if _ram_cache_ready:
  print(f'[Célula 4] Série já carregada na RAM desta sessão ({len(md_rmf.all_sequences_dbz)} sequências).')
else:
  adaptee.build_period_series(start_ym=(2019, 2), end_ym=(2025, 5),
                              months_filter=[2, 3, 4, 5], stride=STRIDE_JANELAS)
t_etl_elapsed_s = time.perf_counter() - t_etl0

# Lote por memória de GPU (a taxa de aprendizado efetiva é mantida em 5e-5 pelo acúmulo).
# O lote efetivo (lote x acúmulo) fica em 16 mesmo com VRAM sobrando: aumentá-lo mudaria a dinâmica do
# otimizador e a taxa de aprendizado efetiva (Célula 4, planilha da taxa de aprendizado).
if torch.cuda.is_available():
  total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
  BATCH_SIZE_AUTO, ACCUM_STEPS_AUTO = (16, 1) if total_vram_gb >= 65 else (4, 4) if total_vram_gb >= 20 else (2, 8)
  # Processos da CPU que preparam os próximos lotes enquanto a GPU calcula o atual (memória fixada e
  # workers persistentes já vêm da Célula 3).
  NUM_WORKERS_AUTO = min(8, os.cpu_count() or 2)
else:
  total_vram_gb, BATCH_SIZE_AUTO, ACCUM_STEPS_AUTO = 0.0, 2, 8
  NUM_WORKERS_AUTO = None

t_split0 = time.perf_counter()
_prev = None
if _ram_cache_ready:
  _prev = (md_rmf.all_sequences_dbz, md_rmf.all_timestamps, md_rmf.target_lats, md_rmf.target_lons)
md_rmf = IMERGToMiaradAdapter(
    adaptee=adaptee, cache_dir=SERIES_CACHE_DIR, seq_len=TOTAL_WINDOW, grid_size=GRID_SIZE, res_km=RES_KM,
    train_norm=(0.0, 60.0, -1.0, 1.0), test_norm=(0.0, 60.0, 0.0, 60.0), batch_size=BATCH_SIZE_AUTO,
    stage_mode='gptcast', env_tag=ENV_TAG, env_label=ENV_LABEL, val_crop=VAL_CROP,
    num_workers=NUM_WORKERS_AUTO,
)
if _prev is not None:
  md_rmf.all_sequences_dbz, md_rmf.all_timestamps, md_rmf.target_lats, md_rmf.target_lons = _prev
  del _prev
dataloaders_rmf = md_rmf.create_gptcast_splits()
t_split_elapsed_s = time.perf_counter() - t_split0

if not torch.cuda.is_available():
  print('\n' + '=' * 78)
  print(f'[Célula 4] Série pronta em CPU ({adaptee._fmt(t_etl_elapsed_s)}); DataLoaders em {t_split_elapsed_s:.2f} s.')
  print('Treino e inferência pausados para não rodar em CPU.')
  print('Próximo passo: "Ambiente de execução" -> "Alterar o tipo de ambiente" -> GPU, e rode as Células 1 a 4 de novo.')
  print('=' * 78)
else:
  device = 'cuda'

  # ------------------------------------------- aceleração na GPU (CUDA / Tensor Cores)
  # O Trainer com precision='bf16-mixed' já roda as operações pesadas em bf16 (autocast). Estas opções
  # cuidam do que sobra: multiplicações de matrizes e convoluções em float32 passam a usar TF32 nos Tensor
  # Cores, e a cuDNN escolhe uma vez o algoritmo de convolução mais rápido (as imagens têm tamanho fixo).
  # Nenhuma delas muda a tokenização, o lote efetivo ou a taxa de aprendizado, então não força novo treino.
  torch.set_float32_matmul_precision('high')
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  torch.backends.cudnn.benchmark = True
  ACELERACAO_CUDA = (f'bf16 (autocast) + TF32 + cudnn.benchmark | {NUM_WORKERS_AUTO} workers, memória fixada | '
                     f'CUDA {torch.version.cuda}')

  def _sync_timer() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()

  # ------------------------------------------- 3. tokenização espaço-temporal (patch)
  # O lote de treino chega como (B, T, H, W). O GPTCast original treina com os T quadros EMPILHADOS na
  # vertical numa imagem (T*H) x W (stack_seq='v' em configs/experiment/gptcast_8x8.yaml) e o forecast()
  # codifica o contexto do mesmo jeito (gptcast.py: "s h w -> (s h) w"). Codificar quadro a quadro dá outros
  # tokens (em 16/11/2026, só 13,7 % iguais), porque as convoluções e a autoatenção do VQGAN enxergam os
  # quadros vizinhos na imagem empilhada. Aqui o treino empilha, igual à previsão. A ordem dos 512 tokens
  # (quadro a quadro, linha a linha) é a mesma nos dois casos.
  _orig_encode_to_z = getattr(GPTCast, '_orig_encode_to_z', GPTCast.encode_to_z)
  GPTCast._orig_encode_to_z = _orig_encode_to_z

  @torch.no_grad()
  def _spatiotemporal_encode_to_z(self, x: torch.Tensor):
    if x.ndim == 4 and x.shape[1] > 1:
      b, t, h, w = x.shape
      x = x.reshape(b, 1, t * h, w).contiguous()      # (B, T, H, W) -> (B, 1, T*H, W), quadro 1 no topo
    quant_z, _, info = self.first_stage_model.encode(x)
    return quant_z, info.view(quant_z.shape[0], -1)

  GPTCast.encode_to_z = _spatiotemporal_encode_to_z

  # ------------------------------------------------------------ 4. registro de tempos
  timing_log_json = os.path.join(RESULTS_ENV_DIR, f'log_tempos_treino_e_inferencia_{ENV_TAG}_2019_2025.json')
  timing_log_csv = timing_log_json.replace('.json', '.csv')
  training_record_json = os.path.join(CHECKPOINTS_ENV_DIR, f'registro_treino_{ENV_TAG}.json')

  # Log anterior: lido só para reconstruir, uma vez, o registro do checkpoint antigo.
  _old_log = {}
  if os.path.exists(timing_log_json):
    try:
      with open(timing_log_json) as f:
        _old_log = json.load(f)
    except Exception:
      _old_log = {}

  # O log é reconstruído a cada sessão: nada de herdar silenciosamente números de sessões anteriores.
  timing_gpu_log = {
      'ambiente_tag': ENV_TAG,
      'gpu_utilizada': torch.cuda.get_device_name(0),
      'vram_total_gb': round(total_vram_gb, 2),
      'aceleracao_cuda': ACELERACAO_CUDA,
      'data_sessao_utc': datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
      'serie_cronologica': '2019-2025 (Treino: 2019-2023 | Val: 2024 | Teste: 2025)',
      'passo_entre_janelas_min': STRIDE_JANELAS * 30,
      'datamodule_split_setup_s': round(t_split_elapsed_s, 3),
      'etl_tempo_sessao_atual_s': round(t_etl_elapsed_s, 2),
  }
  if os.path.exists(adaptee.log_csv_path):
    df_etl = pd.read_csv(adaptee.log_csv_path)
    medidos = df_etl['Tempo_Total_s'].dropna()
    timing_gpu_log['etl_meses_processados'] = int(len(df_etl))
    timing_gpu_log['etl_meses_com_tempo_medido'] = int(len(medidos))
    timing_gpu_log['etl_tempo_acumulado_total_s'] = round(float(medidos.sum()), 2)

  # ------------------------------------------------------------ 5. modelo base (Zenodo)
  t0 = _sync_timer()
  if globals().get('gpt8_zeroshot') is not None:
    gpt8_base = copy.deepcopy(gpt8_zeroshot).to(device)
    timing_gpu_log['origem_modelo_base'] = 'reaproveitado da VRAM desta sessão'
  else:
    gpt8_base = GPTCast.load_from_zenodo('gptcast_8', device=device)
    gpt8_zeroshot = copy.deepcopy(gpt8_base).to(device).eval()
    timing_gpu_log['origem_modelo_base'] = 'Zenodo (gptcast_8)'
    timing_gpu_log['carregamento_modelo_base_zenodo_s'] = round(_sync_timer() - t0, 3)

  def _find_best_checkpoint():
    """Checkpoint registrado pelo último treino; senão, o de menor val/loss no nome; senão, o mais recente.

    Para o checkpoint antigo (sem registro), cria uma vez um registro reconstruído do log de tempos
    anterior, marcado como tal.
    """
    if os.path.exists(training_record_json):
      with open(training_record_json) as f:
        rec = json.load(f)
      p = os.path.join(CHECKPOINTS_ENV_DIR, rec.get('checkpoint', ''))
      if os.path.isfile(p):
        return p, rec
    cands = glob.glob(os.path.join(CHECKPOINTS_ENV_DIR, 'gptcast8_rmf_2km*best*.ckpt'))
    if not cands:
      return None, None
    def _key(p):
      m = re.search(r'best-\d+-([0-9]+\.[0-9]+)\.ckpt$', os.path.basename(p))
      return (float(m.group(1)) if m else np.inf, -os.path.getmtime(p))
    best = sorted(cands, key=_key)[0]
    if _old_log.get('finetuning_tempo_total_s') is not None:
      rec = {k: _old_log[k] for k in ('epocas_executadas', 'melhor_val_loss_2024', 'finetuning_tempo_total_s',
                                      'finetuning_tempo_total_min', 'finetuning_tempo_medio_por_epoca_s',
                                      'precisao_numerica', 'batch_size_treino') if k in _old_log}
      m = re.search(r'epoch=(\d+)', os.path.basename(best))
      rec.update({'checkpoint': os.path.basename(best), 'melhor_epoca': int(m.group(1)) if m else None,
                  'data_treino_utc': 'data não registrada (reconstruído do log de tempos antigo)',
                  'semente_treino': 'não fixada no treino original'})
      with open(training_record_json, 'w') as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
      return best, rec
    return best, None

  def _load_weights(model, path):
    state = torch.load(path, map_location=device)['state_dict']
    res = model.load_state_dict(state, strict=False)
    tr_missing = [k for k in res.missing_keys if k.startswith('transformer.')]
    if tr_missing:
      raise RuntimeError(f'Checkpoint {os.path.basename(path)} não contém {len(tr_missing)} pesos do Transformer '
                         f'(ex.: {tr_missing[0]}). O modelo seria o zero-shot: abortando.')
    if res.missing_keys or res.unexpected_keys:
      print(f'[Aviso Célula 4] Chaves ausentes: {len(res.missing_keys)} | inesperadas: {len(res.unexpected_keys)}.')
    return model

  best_ckpt_path, train_rec = (None, None) if FORCE_RETRAIN else _find_best_checkpoint()
  if best_ckpt_path and (train_rec or {}).get('tokenizacao_treino') != TOKENIZACAO_TREINO:
    if USAR_CHECKPOINT_QUADRO_A_QUADRO:
      print(f'[Aviso Célula 4] Carregando {os.path.basename(best_ckpt_path)}, treinado com tokenização QUADRO A QUADRO, '
            'diferente da usada na previsão. Use só para reproduzir números antigos.')
      train_rec = dict(train_rec or {}, tokenizacao_treino='[OCULTO]')
    else:
      print(f'[Célula 4] O checkpoint {os.path.basename(best_ckpt_path)} foi treinado com tokenização quadro a quadro, '
            'incompatível com a previsão. Um novo fine-tuning (tokenização empilhada) será feito agora; o checkpoint '
            'antigo continua no Drive.')
      best_ckpt_path, train_rec = None, None

  if best_ckpt_path:
    # -------------------------------------------------------- 6a. checkpoint existente
    t0 = _sync_timer()
    gpt8 = _load_weights(gpt8_base, best_ckpt_path).to(device).eval()
    timing_gpu_log['carregamento_checkpoint_drive_s'] = round(_sync_timer() - t0, 3)
    timing_gpu_log['checkpoint_ativo'] = os.path.basename(best_ckpt_path)
    timing_gpu_log['modo_obtencao_modelo'] = f'Checkpoint carregado do Drive ({ENV_TAG})'
    if train_rec:  # números do treino que GEROU este checkpoint, com a data dele
      for k in ('epocas_executadas', 'melhor_epoca', 'melhor_val_loss_2024', 'finetuning_tempo_total_s',
                'finetuning_tempo_total_min', 'finetuning_tempo_medio_por_epoca_s', 'precisao_numerica',
                'batch_size_treino', 'learning_rate_efetiva', 'semente_treino', 'tokenizacao_treino'):
        if k in train_rec:
          timing_gpu_log[k] = train_rec[k]
      timing_gpu_log['finetuning_origem'] = f"treino de {train_rec.get('data_treino_utc', 'data não registrada')}"
    else:
      timing_gpu_log['finetuning_origem'] = 'checkpoint sem registro de treino (tempos e épocas desconhecidos)'
    print(f'[Célula 4] Checkpoint carregado: {os.path.basename(best_ckpt_path)} | {timing_gpu_log["finetuning_origem"]}')
  else:
    # ------------------------------------------------------------- 6b. fine-tuning
    pl_active = pl_modern if issubclass(GPTCast, pl_modern.LightningModule) else pl_legacy
    pl_active.seed_everything(SEED, workers=True)
    precision_mode = 'bf16-mixed' if torch.cuda.is_bf16_supported() else '16-mixed'
    gpt8 = gpt8_base

    # Estágio 1 (VQGAN) congelado; Estágio 2 (GPT-2) treinável
    gpt8.train()
    gpt8.first_stage_model.eval()
    for p in gpt8.first_stage_model.parameters():
      p.requires_grad = False
    for p in gpt8.transformer.parameters():
      p.requires_grad = True

    target_lr = 5e-5  # GPTCast: lr = base_lr x batch x acúmulo x nº de GPUs
    gpt8.base_learning_rate = target_lr / (BATCH_SIZE_AUTO * ACCUM_STEPS_AUTO)
    gpt8.learning_rate = target_lr

    run_tag = datetime.datetime.utcnow().strftime('%Y%m%dT%H%MZ')
    checkpoint_cb = pl_active.callbacks.ModelCheckpoint(
        dirpath=CHECKPOINTS_ENV_DIR,
        filename=f'gptcast8_rmf_2km_{ENV_TAG}_{run_tag}_best-{{epoch:02d}}-{{val/loss:.4f}}',
        auto_insert_metric_name=False, monitor='val/loss', mode='min', save_top_k=1, save_last=True,
    )
    early_stop_cb = pl_active.callbacks.EarlyStopping(monitor='val/loss', patience=4, mode='min', verbose=True)
    csv_logger = pl_active.loggers.CSVLogger(save_dir=RESULTS_ENV_DIR, name=f'curva_treino_{ENV_TAG}', version=run_tag)

    trainer = pl_active.Trainer(
        accelerator='gpu', devices=1, max_epochs=15, precision=precision_mode,
        accumulate_grad_batches=ACCUM_STEPS_AUTO, callbacks=[checkpoint_cb, early_stop_cb],
        logger=csv_logger, log_every_n_steps=5, deterministic=False,
    )
    print(f'[Célula 4] Fine-tuning: lote {BATCH_SIZE_AUTO} x acúmulo {ACCUM_STEPS_AUTO}, {precision_mode}, '
          f'lr {target_lr}, semente {SEED}, validação com corte "{VAL_CROP}".')
    print(f'[Célula 4] Aceleração: {ACELERACAO_CUDA}.')
    torch.cuda.reset_peak_memory_stats(0)
    t0 = _sync_timer()
    trainer.fit(gpt8, datamodule=md_rmf)
    t_fit = _sync_timer() - t0
    pico_vram_treino_gb = round(torch.cuda.max_memory_allocated(0) / 1024**3, 2)
    epochs_ran = int(trainer.current_epoch)

    best_ckpt_path = checkpoint_cb.best_model_path
    if not best_ckpt_path or not os.path.exists(best_ckpt_path):
      raise RuntimeError('O treino terminou sem checkpoint de melhor val/loss.')
    best_val = float(checkpoint_cb.best_model_score.cpu().item())
    m = re.search(r'best-(\d+)-', os.path.basename(best_ckpt_path))
    best_epoch = int(m.group(1)) if m else None
    gpt8 = _load_weights(gpt8, best_ckpt_path).to(device).eval()

    train_rec = {
        'data_treino_utc': run_tag, 'checkpoint': os.path.basename(best_ckpt_path),
        'epocas_executadas': epochs_ran, 'melhor_epoca': best_epoch,
        'melhor_val_loss_2024': round(best_val, 4), 'finetuning_tempo_total_s': round(t_fit, 2),
        'finetuning_tempo_total_min': round(t_fit / 60, 2),
        'finetuning_tempo_medio_por_epoca_s': round(t_fit / max(epochs_ran, 1), 2),
        'precisao_numerica': precision_mode, 'batch_size_treino': BATCH_SIZE_AUTO,
        'acumulo_gradiente': ACCUM_STEPS_AUTO, 'learning_rate_efetiva': target_lr, 'semente_treino': SEED,
        'corte_validacao': VAL_CROP, 'tokenizacao_treino': TOKENIZACAO_TREINO, 'gpu': torch.cuda.get_device_name(0),
        'aceleracao_cuda': ACELERACAO_CUDA, 'pico_vram_treino_gb': pico_vram_treino_gb,
        'curva_por_epoca_csv': os.path.join(csv_logger.log_dir, 'metrics.csv'),
    }
    with open(training_record_json, 'w') as f:
      json.dump(train_rec, f, indent=2, ensure_ascii=False)
    timing_gpu_log.update({k: v for k, v in train_rec.items() if k not in ('checkpoint', 'data_treino_utc')})
    timing_gpu_log['checkpoint_ativo'] = os.path.basename(best_ckpt_path)
    timing_gpu_log['modo_obtencao_modelo'] = f'Fine-tuning executado nesta sessão ({ENV_TAG})'
    timing_gpu_log['finetuning_origem'] = f'esta sessão ({run_tag})'
    print(f'[Célula 4] Fine-tuning concluído: {epochs_ran} épocas em {t_fit / 60:.2f} min | '
          f'melhor val/loss {best_val:.4f} na época {best_epoch} | pico de VRAM {pico_vram_treino_gb} GB '
          f'de {total_vram_gb:.0f} GB | curva em {train_rec["curva_por_epoca_csv"]}')

  # Os dois modelos ficam na GPU e em modo de avaliação para as células seguintes (4.1, 5 a 13.1).
  gpt8 = gpt8.to(device).eval()
  gpt8_zeroshot = gpt8_zeroshot.to(device).eval()

  # ------------------------------------------------------- 7. estudo de caso (validação)
  full_sequence_dbz, dt = md_rmf.get_sequence_by_datetime(CASE_STUDY_ISO)
  input_data_dbz = full_sequence_dbz[:CONTEXT_STEPS]
  ground_truth_dbz = full_sequence_dbz[CONTEXT_STEPS:]
  ground_truth_rain = dbz_to_rainfall(np.clip(ground_truth_dbz.data, 0.0, 60.0))
  clean_in_dbz = np.ma.masked_invalid(np.clip(input_data_dbz, 0.0, 60.0))
  print(f'\n[Célula 4] Janela terminando em {dt} | entrada {input_data_dbz.shape} | verdade {ground_truth_dbz.shape}')

  # --------------------------------------------- 8. inferência (+30 a +120 min), cronometrada
  @torch.inference_mode()                    # sem registro de gradientes: menos memória e menos trabalho na GPU
  def _forecast(model):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    raw = model.forecast(clean_in_dbz, steps=FORECAST_STEPS, units='dbz', verbosity=0)
    raw = raw.filled(0.0) if isinstance(raw, np.ma.MaskedArray) else np.asarray(raw)
    return np.clip(np.nan_to_num(raw, nan=0.0), 0.0, 60.0).astype(np.float32)

  def _timed(model):
    out = _forecast(model)  # aquecimento (kernels CUDA, cache): não entra no tempo
    times = []
    for _ in range(N_TIMING_REPEATS):
      t0 = _sync_timer()
      out = _forecast(model)
      times.append(_sync_timer() - t0)
    return out, float(np.mean(times)), float(np.std(times))

  forecast8_dbz, t_ft, sd_ft = _timed(gpt8)
  forecast8_zs_dbz, t_zs, sd_zs = _timed(gpt8_zeroshot)
  forecast8_rain = dbz_to_rainfall(forecast8_dbz)
  forecast8_zeroshot_rain = dbz_to_rainfall(forecast8_zs_dbz)

  timing_gpu_log.update({
      'inferencia_4passos_finetuned_s': round(t_ft, 4),
      'inferencia_4passos_finetuned_desvio_s': round(sd_ft, 4),
      'inferencia_por_passo_finetuned_s': round(t_ft / FORECAST_STEPS, 4),
      'inferencia_4passos_zeroshot_s': round(t_zs, 4),
      'inferencia_4passos_zeroshot_desvio_s': round(sd_zs, 4),
      'inferencia_por_passo_zeroshot_s': round(t_zs / FORECAST_STEPS, 4),
      'inferencia_repeticoes': N_TIMING_REPEATS,
      'estudo_de_caso': f'{dt:%Y-%m-%d %H:%M} UTC ({md_rmf.partition_of(np.datetime64(dt, "m"))})',
  })

  with open(timing_log_json, 'w') as f:
    json.dump(timing_gpu_log, f, indent=2, ensure_ascii=False, default=str)
  pd.DataFrame([timing_gpu_log]).to_csv(timing_log_csv, index=False)

  print(f'\n=== Célula 4: tempos e origem do modelo ({ENV_LABEL}) ===')
  display(pd.DataFrame([timing_gpu_log]).T.rename(columns={0: f'Valor ({ENV_TAG})'}))
  print(f'Pico previsto (ajustado / zero-shot) a +30 min: {forecast8_rain[0].max():.2f} / '
        f'{forecast8_zeroshot_rain[0].max():.2f} mm/h | observado: {ground_truth_rain[0].max():.2f} mm/h')
  print(f'Relatório salvo em: {timing_log_csv}')
