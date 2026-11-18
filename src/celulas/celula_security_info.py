# @title
# Célula (Security info) : Verificação de Hardware e Unidade de Processamento Ativa (`nvidia-smi`)
# ==============================================================================
# AUDITORIA DE HARDWARE: Verificação de Unidade de Processamento (GPU / CUDA)
# ==============================================================================

import subprocess
import torch

print("=" * 70)
print("RELATÓRIO DE TELEMETRIA DO AMBIENTE DE EXECUÇÃO COLAB")
print("=" * 70)

cuda_status = torch.cuda.is_available()
print(f"[*] CUDA Disponível: {cuda_status}")

if cuda_status:
  device_name = torch.cuda.get_device_name(0)
  total_vram_gb = round(
      torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
  )
  bf16_support = torch.cuda.is_bf16_supported()

  print(f"[*] GPU Atribuída (PyTorch): {device_name}")
  print(f"[*] VRAM Total Dedicada: {total_vram_gb} GiB")
  print(f"[*] Suporte Nativo a BF16: {bf16_support}")
  print("-" * 70)
  print("[*] Relatório Detalhado do Driver (nvidia-smi):")
  try:
    subprocess.run(["nvidia-smi"], check=True)
  except Exception as e:
    print(f"[AVISO] Não foi possível executar o nvidia-smi: {e}")
else:
  print(
      "[!] ATENÇÃO: Nenhuma GPU detectada. O ambiente está rodando apenas em"
      " CPU."
  )
print("=" * 70)
