# @title
# Célula (Machine id): Detecção Automática do Ambiente de Execução - Identidade de Hardware e Isolamento de Artefatos na Memória (`ENV_TAG`)
import torch

if torch.cuda.is_available():
  dev_name = torch.cuda.get_device_name(0)
  vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

  if 'H100' in dev_name:
    ENV_TAG = 'H100_Hopper_80GB'
    ENV_LABEL = 'NVIDIA H100 (Hopper 80 GB)'
  elif '6000' in dev_name or 'Blackwell' in dev_name:
    ENV_TAG = 'G4_RTX6000_Blackwell_95GB'
    ENV_LABEL = 'NVIDIA RTX PRO 6000 Blackwell (95 GB)'
  else:
    ENV_TAG = dev_name.replace(' ', '_')
    ENV_LABEL = dev_name
else:
  ENV_TAG = 'CPU_Padrao'
  ENV_LABEL = 'CPU Padrão'

print(f'[Rastreabilidade Ativa] Ambiente detectado: {ENV_LABEL}')
print(f'[Rastreabilidade Ativa] Tag de identificação: {ENV_TAG}')
