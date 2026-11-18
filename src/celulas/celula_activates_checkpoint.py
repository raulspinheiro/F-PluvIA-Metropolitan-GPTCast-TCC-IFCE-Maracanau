# @title
# Célula (Activates Checkpoint): Conferência do Checkpoint Ativo, Auditoria de Parâmetros Medida no Modelo e
# Quadro-Resumo de Partições e Tempos.
import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch

ENV_TAG = globals().get('ENV_TAG', 'Ambiente_Ativo')
ENV_LABEL = globals().get('ENV_LABEL', 'Hardware Ativo')
CHECKPOINTS_ENV_DIR = globals().get('CHECKPOINTS_ENV_DIR', os.path.join(CHECKPOINTS_DIR, ENV_TAG))
RESULTS_ENV_DIR = globals().get('RESULTS_ENV_DIR', os.path.join(SERIES_CACHE_DIR, f'resultados_{ENV_TAG}'))
os.makedirs(RESULTS_ENV_DIR, exist_ok=True)
CSV_LOG_PATH = os.path.join(SERIES_CACHE_DIR, 'log_tempos_etl_imerg_2019_2025.csv')
GPU_LOG_JSON = os.path.join(RESULTS_ENV_DIR, f'log_tempos_treino_e_inferencia_{ENV_TAG}_2019_2025.json')
GPU_LOG_CSV = GPU_LOG_JSON.replace('.json', '.csv')


def fmt_seconds(sec) -> str:
  if sec is None or (isinstance(sec, float) and not np.isfinite(sec)):
    return 'não registrado'
  m, s = divmod(float(sec), 60)
  h, m = divmod(int(m), 60)
  return f'{h:02d}h:{m:02d}m:{s:05.2f}s'


def fmt_params(n: int) -> str:
  return f'{n:,}'.replace(',', '.') + f' ({n / 1e6:.2f} M)'


def _transformer_digest(state: dict) -> str:
  """Impressão digital (SHA-256) dos pesos do Transformer, independente do dispositivo."""
  h = hashlib.sha256()
  for k in sorted(k for k in state if k.startswith('transformer.')):
    h.update(k.encode())
    h.update(state[k].detach().float().cpu().numpy().tobytes())
  return h.hexdigest()[:16]


if 'gpt8' not in globals() or 'best_ckpt_path' not in globals():
  raise RuntimeError('Rode a Célula 4 antes: gpt8 e best_ckpt_path não existem nesta sessão.')

# ------------------------------------------------- 1. o gpt8 em memória é o checkpoint ativo?
ckpt_state = torch.load(best_ckpt_path, map_location='cpu')['state_dict']
dig_ckpt = _transformer_digest(ckpt_state)
dig_mem = _transformer_digest(gpt8.state_dict())
dig_zs = _transformer_digest(gpt8_zeroshot.state_dict()) if globals().get('gpt8_zeroshot') is not None else None
del ckpt_state

if dig_mem != dig_ckpt:
  raise RuntimeError(f'O gpt8 em memória não corresponde a {os.path.basename(best_ckpt_path)} '
                     f'(impressões {dig_mem} x {dig_ckpt}). Rode a Célula 4 de novo.')
if dig_zs is not None and dig_zs == dig_mem:
  raise RuntimeError('O gpt8 tem os mesmos pesos do zero-shot: o ajuste fino não foi aplicado.')
print(f'[Célula Activates Checkpoint] gpt8 confere com {os.path.basename(best_ckpt_path)} '
      f'(impressão do Transformer {dig_mem}; zero-shot {dig_zs}).')

# ------------------------------------------------------- 2. parâmetros medidos no modelo
n_total = sum(p.numel() for p in gpt8.parameters())
n_vqgan = sum(p.numel() for p in gpt8.first_stage_model.parameters())
n_gpt = sum(p.numel() for p in gpt8.transformer.parameters())
n_other = n_total - n_vqgan - n_gpt
n_mod_vqgan = sum(1 for _ in gpt8.first_stage_model.modules())
n_mod_gpt = sum(1 for _ in gpt8.transformer.modules())

def _row(nome, status, n, n_mod=None):
  return {
      'Ambiente de Execução': ENV_LABEL, 'Hardware_Tag': ENV_TAG,
      'Componente da Arquitetura GPTCast-8x8': nome, 'Status no Fine-Tuning': status,
      'Quantidade de Parâmetros': fmt_params(n), 'Proporção (%)': f'{100 * n / n_total:.2f}%',
      'Módulos': n_mod if n_mod is not None else '',
      'Peso em Memória (FP32 / BF16)': f'{n * 4 / 1024**2:.2f} MB / {n * 2 / 1024**2:.2f} MB',
  }

rows = [
    _row('Modelo completo (Estágio 1 + Estágio 2)', 'VQGAN congelado + GPT-2 ajustado', n_total),
    _row(' -> Estágio 2: Transformer temporal (GPT-2)', 'Ajustado na RMF (treino 2019–2023, validação 2024)', n_gpt, n_mod_gpt),
    _row(' -> Estágio 1: Autoencoder espacial (VQGAN)', 'Congelado (pesos originais do GPTCast)', n_vqgan, n_mod_vqgan),
]
if n_other:
  rows.append(_row(' -> Outros parâmetros', 'conforme o GPTCast', n_other))
df_params = pd.DataFrame(rows)
df_params.to_csv(os.path.join(RESULTS_ENV_DIR, f'arquitetura_parametros_gptcast8_{ENV_TAG}.csv'), index=False)
print(f'\n=== Parâmetros do GPTCast-8x8 ({ENV_LABEL}) ===')
display(df_params.drop(columns=['Hardware_Tag']))

# ------------------------------------------------------------ 3. log da sessão (acréscimo)
gpu_log = dict(globals().get('timing_gpu_log') or {})
if not gpu_log and os.path.exists(GPU_LOG_JSON):
  with open(GPU_LOG_JSON) as f:
    gpu_log = json.load(f)
gpu_log.update({
    'parametros_totais_modelo': int(n_total), 'parametros_treinados_transformer': int(n_gpt),
    'parametros_congelados_vqgan': int(n_vqgan),
    'proporcao_parametros_treinados_pct': round(100 * n_gpt / n_total, 2),
    'impressao_transformer_sha256_16': dig_mem,
})
timing_gpu_log = gpu_log
with open(GPU_LOG_JSON, 'w') as f:
  json.dump(gpu_log, f, indent=2, ensure_ascii=False, default=str)
pd.DataFrame([gpu_log]).to_csv(GPU_LOG_CSV, index=False)

# --------------------------------------------- 4. quadro-resumo: partições e tempos registrados
resumo = [('Hardware / ambiente de execução', ENV_LABEL, f'Tag: {ENV_TAG}'),
          ('Checkpoint ativo', os.path.basename(best_ckpt_path), gpu_log.get('finetuning_origem', ''))]

split = getattr(globals().get('md_rmf'), 'split_summary', None)
if split is not None:  # contagens do conjunto efetivamente carregado (Célula 3)
  total = int(split['Sequencias'].sum())
  resumo.append(('Sequências de 11 quadros a 2 km', f'{total}',
                 f"passo de {gpu_log.get('passo_entre_janelas_min', '?')} min entre janelas"))
  for _, r in split.iterrows():
    resumo.append((f" -> {r['Particao']}", f"{int(r['Sequencias'])} ({r['Proporcao_pct']}%)",
                   f"{int(r['Meses'])} meses | {r['Primeira_janela']} a {r['Ultima_janela']}"))

if os.path.exists(CSV_LOG_PATH):
  df_etl = pd.read_csv(CSV_LOG_PATH)  # só leitura: a Célula 2 é a dona deste arquivo
  medidos = df_etl['Tempo_Total_s'].dropna()
  resumo.append(('ETL medido (download + interpolação a 2 km)', f'{medidos.sum():.2f} s',
                 f'{fmt_seconds(medidos.sum())} em {len(medidos)} de {len(df_etl)} meses'))

ft_s = gpu_log.get('finetuning_tempo_total_s')
ep = gpu_log.get('epocas_executadas')
resumo.append(('Fine-tuning na GPU',
               f'{ft_s:.2f} s' if isinstance(ft_s, (int, float)) else 'não registrado',
               f"{fmt_seconds(ft_s)} | {ep if ep is not None else '?'} épocas | melhor época "
               f"{gpu_log.get('melhor_epoca', '?')} | val/loss {gpu_log.get('melhor_val_loss_2024', '?')}"))
inf_ft, inf_zs = gpu_log.get('inferencia_4passos_finetuned_s'), gpu_log.get('inferencia_4passos_zeroshot_s')
if inf_ft is not None and inf_zs is not None:
  resumo.append(('Inferência 4 passos (ajustado / zero-shot)', f'{inf_ft:.4f} s / {inf_zs:.4f} s',
                 f"média de {gpu_log.get('inferencia_repeticoes', 1)} repetições | "
                 f'{inf_ft / 4:.4f} / {inf_zs / 4:.4f} s por passo'))

df_resumo = pd.DataFrame(resumo, columns=['Métrica Consolidada (Série 2019–2025 IMERG-FR)',
                                          'Valor Numérico', 'Tempo Formatado / Proporção'])
df_resumo.insert(0, 'Ambiente_Tag', ENV_TAG)
df_resumo.to_csv(os.path.join(RESULTS_ENV_DIR, f'resumo_dataset_e_tempos_{ENV_TAG}_2019_2025.csv'), index=False)
print(f'\n=== Quadro-resumo: partições e tempos ({ENV_LABEL}) ===')
display(df_resumo.drop(columns=['Ambiente_Tag']))

if os.path.exists(CSV_LOG_PATH):
  print(f'\n=== Log de ETL por mês ({os.path.basename(CSV_LOG_PATH)}) ===')
  pd.set_option('display.max_rows', 35)
  cols = [c for c in ('Ano_Mes', 'Origem', 'Sequencias_Extraidas', 'Tempo_Download_s',
                      'Tempo_Processamento_2km_s', 'Tempo_Total_Formatado', 'Passos_Obtidos',
                      'Passos_Esperados', 'Metodo_Interp') if c in df_etl.columns]
  display(df_etl[cols])
