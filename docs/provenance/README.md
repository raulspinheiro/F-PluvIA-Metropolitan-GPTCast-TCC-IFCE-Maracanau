# Auditoria Científica, Proveniência Forense e Rastreabilidade de Hardware (`/docs/provenance`)

Este diretório armazena os manifestos criptográficos, relatórios de inspeção visual, carimbos de tempo e metadados físicos de execução que compõem o **Protocolo de Auditoria e Proveniência Científica** do projeto **F-PluvIA Metropolitan**, desenvolvido como Trabalho de Conclusão de Curso (TCC) no formato de **Artigo Científico** para o Bacharelado em Ciência da Computação do Instituto Federal de Educação, Ciência e Tecnologia do Ceará (IFCE), Campus Maracanaú (semestre letivo `2026.2`).

---

## 1. Finalidade Metodológica e Reprodutibilidade

Experimentos de grande porte em Inteligência Artificial Generativa e modelagem atmosférica (*nowcasting*) executados em ambientes de computação em nuvem (*cloud compute* como o **Google Colab Pro+**) exigem garantias formais de que os resultados numéricos, painéis cartográficos e pesos ajustados derivam estritamente do código submetido e dos aceleradores de hardware declarados.

Os manifestos contidos nesta pasta vinculam, de maneira auditável e imutável:
* O código-fonte integral do notebook experimental com nomenclatura canônica preservada (`notebooks/Model_Annual_Full_Training_Series_TCC2_-_2026_2_-_RAUL_SANTIAGO_PINHEIRO.ipynb`);
* O ambiente operacional do container Linux (`boot_id`) e a telemetria física das GPUs utilizadas;
* Os artefatos numéricos (`.csv`) e visuais a $300\text{ dpi}$ (`.png`/`.pdf`) gerados durante a execução das Células 1 a 13.1;
* A autoria acadêmica e filiação institucional perante o IFCE Campus Maracanaú.

---

## 2. Camadas de Auditoria Forense (Célula Extra — Carimbo Digital)

Cada certificado emitido ao término da execução opera sobre quatro eixos de validação:

### 2.1 Identidade da Instância e Ambiente de Execução
* **Linux Kernel Boot ID:** Leitura direta de `/proc/sys/kernel/random/boot_id`, garantindo o identificador exclusivo da sessão efêmera do nó de execução.
* **Carimbos Temporais Sincronizados:** Registro simultâneo em tempo universal coordenado (`UTC`) e no fuso horário local de Fortaleza (`America/Fortaleza`, UTC-3).
* **Versões de Software:** Rastreamento exato das versões de Python, PyTorch e CUDA Toolkit ativas no momento da inferência.

### 2.2 Telemetria Física de Hardware (GPUs NVIDIA)
* **GPU UUID:** Identificador físico unívoco extraído via `nvidia-smi --query-gpu=uuid` da placa alocada.
* **Arquiteturas Auditadas no Google Colab Pro+:**
  * **NVIDIA RTX 6000 Blackwell (G4):** $95\text{ GB}$ GDDR7, suporte nativo a precisão mista `BF16`.
  * **NVIDIA H100 Hopper:** $80\text{ GB}$ HBM3, barramento de alta largura de banda.
* **Métricas de Consumo:** Monitoramento de VRAM total e pico de memória alocada (`torch.cuda.max_memory_allocated`), versão do driver gráfico e validação de suporte a ponto flutuante `BF16`.

### 2.3 Inventário Criptográfico de Artefatos (`SHA-256`)
Todos os produtos gerados pelo pipeline são submetidos a cálculo recursivo de *hash* criptográfico de 256 bits (`hashlib.sha256`):
* **Notebook Experimental (`.ipynb`):** Assinatura binária do estado fechado do notebook no Google Drive antes do envio ao GitHub.
* **Modelos e Checkpoints (`.pt`/`.ckpt`):** Estados dos pesos do Transformer autorregressivo GPT-2 ($\text{val/loss} = 4{,}822$) com VQGAN espacial congelado (auditados em manifesto mesmo quando retidos no Drive por excederem o limite de $95\text{ MB}$ do GitHub).
* **Tabelas de Resultados (`.csv`):** Relatórios consolidados de métricas contínuas (MAE, RMSE, SSIM, PCC) e categóricas (CSI, FAR, POD, F1-Score, HSS, ETS) para $+30$, $+60$, $+90$ e $+120\text{ min}$.
* **Artefatos Visuais a 300 dpi (`.png`/`.pdf`):** Painéis de precipitação na RMF, espectros **RAPSD** (escala de Nyquist em $4\text{ km}$), diagramas tridimensionais **SAL** e curvas multiescala **FSS** ($2\text{ km}$ a $34\text{ km}$).

### 2.4 Assinatura Mestre da Sessão (`Master Session Hash`)
Um *hash* raiz unificado gerado pela concatenação determinística dos identificadores do autor, instituição, `boot_id`, UUID da GPU e os *hashes* individuais ordenados de todos os arquivos produzidos. Qualquer modificação posterior em um único bit de qualquer artefato invalida a assinatura mestre.

---

## 3. Padrão de Nomenclatura e Formatos Emitidos

Para cada sessão executada no Google Colab Pro+, o gerador emite um par de arquivos sincronizados (JSON para validação programática e Markdown para leitura imediata na interface web do GitHub):

```text
docs/provenance/
├── README.md                                          # Este guia do protocolo de auditoria e proveniência
├── certificado_autoria_NVIDIA_RTX_6000_*.json         # Manifesto estruturado JSON (Arquitetura Blackwell G4)
├── certificado_autoria_NVIDIA_RTX_6000_*.md           # Relatório visual com tabela SHA-256 (Arquitetura Blackwell G4)
├── certificado_autoria_NVIDIA_H100_*.json             # Manifesto estruturado JSON (Arquitetura Hopper H100)
└── certificado_autoria_NVIDIA_H100_*.md               # Relatório visual com tabela SHA-256 (Arquitetura Hopper H100)
```

---

## 4. Como Verificar a Integridade de um Artefato Localmente

Para auditar qualquer arquivo clonado deste repositório e comprovar que ele corresponde exatamente ao artefato gerado na GPU do Google Colab Pro+, execute no terminal:

```bash
# No Linux / macOS (Bash):
sha256sum notebooks/Model_Annual_Full_Training_Series_TCC2_-_2026_2_-_RAUL_SANTIAGO_PINHEIRO.ipynb
sha256sum results/tables/*.csv

# No Windows (PowerShell):
Get-FileHash -Algorithm SHA256 notebooks/Model_Annual_Full_Training_Series_TCC2_-_2026_2_-_RAUL_SANTIAGO_PINHEIRO.ipynb
Get-FileHash -Algorithm SHA256 results/tables/*.csv
```

Compare o código hexadecimal de 64 caracteres retornado no terminal com a coluna **Hash SHA-256** do arquivo `certificado_autoria_*.md` (ou campo `"sha256"` em `certificado_autoria_*.json`).

---

## Navegação Cruzada

* [Voltar ao Guia do Diretório `/docs` (`../structure.md`)](../structure.md)
* [Voltar ao README Principal do Projeto (`../../README.md`)](../../README.md)
