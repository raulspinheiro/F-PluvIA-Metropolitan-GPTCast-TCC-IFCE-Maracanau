# Auditoria Científica, Proveniência Forense e Rastreabilidade de Hardware

Este diretório armazena os manifestos criptográficos, carimbos de tempo e metadados de execução que compõem o **Protocolo de Auditoria e Proveniência Científica** do projeto **F-PluvIA Metropolitan**, desenvolvido como Trabalho de Conclusão de Curso (TCC) no formato de **Artigo Científico** para o Bacharelado em Ciência da Computação do Instituto Federal de Educação, Ciência e Tecnologia do Ceará (IFCE), Campus Maracanaú (semestre letivo 2026.2).

---

## 1. Finalidade Metodológica e Reprodutibilidade

Experimentos de grande porte em Inteligência Artificial Generativa e modelagem atmosférica (*nowcasting*) executados em ambientes de computação em nuvem (*cloud compute* como o **Google Colab Pro+**) exigem garantias formais de que os resultados numéricos, gráficos e pesos ajustados derivam estritamente do código submetido e dos aceleradores de hardware declarados.

Os manifestos contidos nesta pasta vinculam, de maneira auditável e imutável:
* O código-fonte integral do notebook experimental (`Model_Annual_Full_Training_Series_TCC2_-_2026_2_-_RAUL_SANTIAGO_PINHEIRO.ipynb`);
* O ambiente operacional do container Linux e a telemetria física das GPUs utilizadas;
* Os artefatos numéricos e cartográficos gerados durante a execução das Células 1 a 13.1;
* A autoria acadêmica e filiação institucional perante o IFCE Campus Maracanaú.

---

## 2. Camadas de Auditoria Forense

Cada certificado emitido pela Célula Extra (Carimbo Digital) opera sobre quatro eixos de validação:

### 2.1 Identidade da Instância e Ambiente de Execução
* **Linux Kernel Boot ID:** Leitura direta de `/proc/sys/kernel/random/boot_id`, garantindo o identificador exclusivo da sessão efêmera do nó de execução.
* **Carimbos Temporais Sincronizados:** Registro em tempo universal coordenado (`UTC`) e no horário padrão de Brasília/Fortaleza (`America/Fortaleza`, UTC-3).
* **Versões de Software:** Rastreamento exato das versões de Python, PyTorch e CUDA Toolkit ativas no momento da convergência.

### 2.2 Telemetria Física de Hardware (GPUs NVIDIA)
* **GPU UUID:** Identificador físico unívoco extraído via `nvidia-smi` da placa alocada.
* **Arquiteturas Comparadas:** Discriminação entre as execuções de treinamento e inferência no Google Colab Pro+:
  * **NVIDIA RTX 6000 Blackwell (G4):** 95 GB GDDR7, precisão nativa `BF16`.
  * **NVIDIA H100 Hopper:** 80 GB HBM3, barramento SXM5/PCIe.
* **Métricas de Consumo:** Monitoramento de pico de VRAM alocada (`torch.cuda.max_memory_allocated`), versão do driver gráfico e suporte a ponto flutuante de 16 bits.

### 2.3 Inventário Criptográfico de Artefatos (`SHA-256`)
Todos os produtos gerados pelo pipeline são submetidos a cálculo recursivo de *hash* criptográfico de 256 bits (`hashlib.sha256`):
* **Modelos e Checkpoints:** Estados dos pesos do Transformer autorregressivo GPT-2 (`val/loss = 4.822`) com VQGAN espacial congelado.
* **Tabelas de Resultados:** Relatórios em `.csv` das métricas de regressão (MAE, RMSE, SSIM, PCC) e decisão (CSI, FAR, POD, HSS, ETS) para os horizontes $+30$, $+60$, $+90$ e $+120\text{ min}$.
* **Artefatos Visuais a 300 dpi:** Mapas de precipitação na RMF, espectros **RAPSD** (escala de Nyquist em 4 km), diagramas tridimensionais **SAL** e curvas de fração multiescala **FSS** (2 km a 34 km).

### 2.4 Assinatura Mestre da Sessão (`Master Session Hash`)
Um *hash* raiz unificado gerado pela concatenação determinística dos identificadores do autor, instituição, `boot_id`, UUID da GPU e os *hashes* individuais de todos os arquivos produzidos. Qualquer alteração posterior em um único bit de qualquer arquivo invalida a assinatura mestre.

---

## 3. Padrão de Nomenclatura dos Arquivos de Certificação

Os arquivos gerados automaticamente pela Célula Extra seguem o padrão estruturado:

```text
docs/provenance/
├── README.md                                      # Este guia de auditoria e proveniência
├── certificado_autoria_NVIDIA_RTX_6000_*.json     # Manifesto estruturado da execução na arquitetura Blackwell (G4)
└── certificado_autoria_NVIDIA_H100_*.json         # Manifesto estruturado da execução na arquitetura Hopper (H100)
