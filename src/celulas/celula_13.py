# @title
# Célula 13: Comparativo de Hardware e Custo no Colab — somente valores MEDIDOS em cada ambiente
# (resultados_<ENV_TAG>/). Sem valores de reserva inventados; a comparação só é feita quando houver
# pelo menos dois ambientes com resultados.
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COMPARATIVE_OUT_DIR = os.path.join(SERIES_CACHE_DIR, 'comparativo_hardware_G4_vs_H100')
os.makedirs(COMPARATIVE_OUT_DIR, exist_ok=True)

# Taxa de consumo de cada GPU em unidades de computação (CU) por hora, como aparece no painel
# "Recursos" do Colab no momento do uso. O Google pode mudar essas taxas: preencha com o que o
# painel mostrou na sua sessão. Ambiente sem taxa informada fica sem custo calculado.
CU_POR_HORA = {
    'G4_RTX6000_Blackwell_95GB': 7.7,     # confirme no painel "Recursos" do Colab; troque se a taxa mudar
    'H100_Hopper_80GB': None,
}


def _latest(folder, pattern):
  files = glob.glob(os.path.join(folder, pattern))
  return max(files, key=os.path.getmtime) if files else None


# Rótulos com a decodificação de cada GPTCast, lida do sufixo da tabela de velocidade da Célula 12.
def _rotulo(modelo, arquivo):
  esc = '_amostragem2024' in os.path.basename(arquivo or '')
  return {'GPTCast-8x8 (Fine-Tuned RMF)': f"GPTCast-8x8 (Fine-Tuned RMF, {'gulosa' if esc else 'sorteio padrão'})",
          'GPTCast-8x8 (Zero-Shot)': f"GPTCast-8x8 (Zero-Shot, {'top-k 10' if esc else 'sorteio padrão'})"}.get(modelo, modelo)


def _read_json(path):
  try:
    with open(path) as f:
      return json.load(f)
  except Exception:
    return {}


rows, speed_rows = [], []
for folder in sorted(glob.glob(os.path.join(SERIES_CACHE_DIR, 'resultados_*'))):
  tag = os.path.basename(folder).replace('resultados_', '')
  log = _read_json(_latest(folder, 'log_tempos_treino_e_inferencia_*.json') or '')
  treino = _read_json(os.path.join(CHECKPOINTS_DIR, tag, f'registro_treino_{tag}.json'))
  if not log:
    continue

  vq_ms = np.nan
  f_vq = _latest(folder, 'tabela3_estagio1_vqgan_*.csv')
  if f_vq:
    d = pd.read_csv(f_vq)
    if 'ms por Quadro (↓)' in d.columns:
      vq_ms = float(d['ms por Quadro (↓)'].iloc[-1])

  f_spd = _latest(folder, 'velocidade_inferencia_teste2025_*.csv')   # o mais recente (Célula 12)
  spd = pd.read_csv(f_spd) if f_spd else pd.DataFrame()
  for _, r in spd.iterrows():
    speed_rows.append({'Hardware_Tag': tag, 'Modelo': _rotulo(r['Modelo'], f_spd), 'Amostras': int(r['Amostras_Teste_2025']),
                       'Tempo médio 4 passos [s]': float(r['Tempo_Medio_4Passos_s']),
                       'Desvio [s]': float(r.get('Desvio_Padrao_4Passos_s', np.nan)),
                       'Dispositivo': r.get('Dispositivo', 'GPU' if 'GPTCast' in str(r['Modelo']) else 'CPU'),
                       'Arquivo': os.path.basename(f_spd)})
  t_ft_inf = next((s['Tempo médio 4 passos [s]'] for s in speed_rows
                   if s['Hardware_Tag'] == tag and 'Fine-Tuned' in s['Modelo'] and 'média' not in s['Modelo']),
                  log.get('inferencia_4passos_finetuned_s', np.nan))

  ft_s = treino.get('finetuning_tempo_total_s', log.get('finetuning_tempo_total_s'))
  ft_s = float(ft_s) if ft_s is not None else np.nan
  epocas = treino.get('epocas_executadas', log.get('epocas_executadas'))
  cu_h = CU_POR_HORA.get(tag)
  rows.append({
      'Hardware_Tag': tag,
      'GPU (medida)': log.get('gpu_utilizada', tag),
      'VRAM medida [GB]': log.get('vram_total_gb', np.nan),
      'Origem do treino': treino.get('data_treino_utc', log.get('finetuning_origem', 'não registrado')),
      'Épocas': epocas if epocas is not None else 'não registrado',
      'Lote / precisão': f"{treino.get('batch_size_treino', log.get('batch_size_treino', '?'))} / "
                         f"{treino.get('precisao_numerica', log.get('precisao_numerica', '?'))}",
      'Fine-tuning [min] (↓)': round(ft_s / 60, 2) if np.isfinite(ft_s) else np.nan,
      'Tempo por época [s] (↓)': round(ft_s / int(epocas), 1) if np.isfinite(ft_s) and isinstance(epocas, (int, float)) and epocas else np.nan,
      'Montagem do DataModule [s]': log.get('datamodule_split_setup_s', np.nan),
      'VQGAN [ms/quadro] (↓)': vq_ms,
      'GPTCast ajustado, 4 passos [s] (↓)': round(float(t_ft_inf), 4) if t_ft_inf is not None else np.nan,
      'Taxa informada [CU/h]': cu_h if cu_h is not None else 'não informada',
      'Custo do fine-tuning [CU]': round(ft_s / 3600 * cu_h, 2) if cu_h and np.isfinite(ft_s) else np.nan,
      'Custo de 1.000 previsões [CU]': round(float(t_ft_inf) * 1000 / 3600 * cu_h, 2) if cu_h and t_ft_inf else np.nan,
  })

df_hw_comp = pd.DataFrame(rows)
df_speed = pd.DataFrame(speed_rows)
df_hw_comp.to_csv(os.path.join(COMPARATIVE_OUT_DIR, 'tabela_comparativa_hardware_G4_vs_H100_2019_2025.csv'), index=False)
if len(df_speed):
  df_speed.to_csv(os.path.join(COMPARATIVE_OUT_DIR, 'comparativo_velocidade_inferencia_modelos_G4_vs_H100.csv'), index=False)

print(f'=== Hardware: {len(df_hw_comp)} ambiente(s) com resultados em {SERIES_CACHE_DIR} ===')
if len(df_hw_comp) == 0:
  print('Nenhum ambiente com log da Célula 4. Rode as Células 1 a 12 primeiro.')
else:
  display(df_hw_comp.set_index('Hardware_Tag').T)
  if len(df_speed):
    print('\n=== Tempo de inferência por modelo no teste de 2025 (Célula 12) ===')
    display(df_speed.set_index(['Hardware_Tag', 'Modelo']))

if len(df_hw_comp) < 2:
  print('\n[Célula 13] Só há um ambiente: não existe comparação entre GPUs. Para comparar com a H100, rode as '
        'Células 1 a 12 num ambiente H100 (o ENV_TAG separa os resultados). Até lá, o artigo não deve citar a H100.')
else:
  base = df_hw_comp.iloc[0]
  df_hw_comp['Aceleração do fine-tuning vs. ' + base['Hardware_Tag']] = (base['Fine-tuning [min] (↓)'] / df_hw_comp['Fine-tuning [min] (↓)']).round(2)
  df_hw_comp['Aceleração da inferência vs. ' + base['Hardware_Tag']] = (base['GPTCast ajustado, 4 passos [s] (↓)'] / df_hw_comp['GPTCast ajustado, 4 passos [s] (↓)']).round(2)
  df_hw_comp.to_csv(os.path.join(COMPARATIVE_OUT_DIR, 'tabela_comparativa_hardware_G4_vs_H100_2019_2025.csv'), index=False)
  print('\n=== Aceleração relativa ===')
  display(df_hw_comp.set_index('Hardware_Tag').iloc[:, -2:])

# ----------------------------------------------------------------------- painel
if len(df_hw_comp):
  labels = df_hw_comp['Hardware_Tag'].tolist()
  x = np.arange(len(labels))
  fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), constrained_layout=True)
  axes[0].bar(x, df_hw_comp['Fine-tuning [min] (↓)'], color='#1f77b4', edgecolor='k', width=0.5)
  axes[0].set_title('Fine-tuning (min, ↓)', fontweight='bold')
  if len(df_speed):
    modelos = df_speed['Modelo'].unique()
    w = 0.8 / len(modelos)
    for k, m in enumerate(modelos):
      vals = [df_speed[(df_speed['Hardware_Tag'] == t) & (df_speed['Modelo'] == m)]['Tempo médio 4 passos [s]'].mean() for t in labels]
      axes[1].bar(x - 0.4 + w * (k + 0.5), vals, width=w, label=m, edgecolor='k')
    axes[1].set_yscale('log')
    # Legenda abaixo do painel, para não cobrir as barras.
    axes[1].legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
  axes[1].set_title('Inferência de 4 passos por janela (s, escala log, ↓)', fontweight='bold')
  custos = pd.to_numeric(df_hw_comp['Custo do fine-tuning [CU]'], errors='coerce')
  axes[2].bar(x, custos, color='#2ca02c', edgecolor='k', width=0.5)
  axes[2].set_title('Custo do fine-tuning (CU, ↓)' + ('' if custos.notna().any() else ' — taxa não informada'), fontweight='bold')
  for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(True, axis='y', ls=':', alpha=0.6)
  fig.suptitle(f'Hardware e custo — {len(labels)} ambiente(s) medido(s)', fontsize=12.5, fontweight='bold')
  path = os.path.join(COMPARATIVE_OUT_DIR, 'painel_benchmark_hardware_G4_vs_H100.png')
  fig.savefig(path, dpi=300, bbox_inches='tight')
  plt.show()
  print(f'[Célula 13] Arquivos em {COMPARATIVE_OUT_DIR}')
