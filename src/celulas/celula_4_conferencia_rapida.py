# Célula 4 (conferência rápida): registro de treino e perda por época
import json, pandas as pd
rec = json.load(open(training_record_json))
print({k: rec.get(k) for k in ('epocas_executadas', 'melhor_epoca', 'melhor_val_loss_2024', 'tokenizacao_treino')})
m = pd.read_csv(rec['curva_por_epoca_csv'])
print(m.groupby('epoch')[['train/loss_epoch', 'val/loss_epoch']].mean().round(4))
