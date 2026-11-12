# Diretório de Notebooks Experimentais (/notebooks)

Este diretório armazena o notebook central de execução e experimentação do projeto **F-PluvIA Metropolitan**, desenvolvido como Trabalho de Conclusão de Curso (TCC) no formato de **Artigo Científico** vinculado ao Bacharelado em Ciência da Computação do Instituto Federal de Educação, Ciência e Tecnologia do Ceará (IFCE), Campus Maracanaú (semestre letivo 2026.2).

---

## Arquivo Principal da Execução

* **Nome do Arquivo:** `Model_Annual_Full_Training_Series_TCC2_-_2026_2_-_RAUL_SANTIAGO_PINHEIRO.ipynb`
* **Ambiente Homologado:** Google Colab Pro+ (Aceleradores de Hardware: NVIDIA RTX 6000 Blackwell / G4 — 95 GB GDDR7 vs. NVIDIA H100 Hopper — 80 GB HBM3).
* **Escopo Experimental:** Pipeline unificado cobrindo desde a aquisição e pré-processamento dos dados brutos até a inferência, validação estatística e emissão de certificados forenses.

---

## Resumo das Etapas do Pipeline (Células 1 a 13.1)

O caderno experimental está estruturado em módulos sequenciais que asseguram a reprodutibilidade integral do estudo:

1. **Ingestão e Pré-processamento Geoespacial (GPM IMERG-FR v7):**
   * Coleta de 28 meses de dados históricos de precipitação por satélite sobre a Região Metropolitana de Fortaleza (RMF — 19 municípios).
   * Interpolação bicúbica para grade métrica contínua de $2\text{ km}$ ($128 \times 128\text{ pixels}$ / $256 \times 256\text{ km}$) com gerenciamento de memória em modo *Zero-Copy*.
2. **Particionamento Temporal Estrito (*Anti-Data Leakage*):**
   * **Treinamento (2019–2023):** 20 meses de quadras chuvosas locais para ajuste do Transformer em precisão mista `BF16`.
   * **Validação (2024):** 4 meses completos (incluindo o evento severo de 18/03/2024) com critério de parada antecipada (*Early Stopping*).
   * **Teste Cego (2025):** 1.729 sequências temporais independentes para verificação fora da amostra.
3. **Ajuste Fino (*Fine-Tuning*) do GPTCast-8x8:**
   * **VQGAN Espacial (90,71 M parâmetros):** Codificador convolucional e livro de códigos discreto ($1.024\text{ tokens}$, $8 \times 8\text{ grade latente}$) mantido congelado.
   * **Transformer Temporal (304,93 M parâmetros):** Rede autorregressiva baseada em GPT-2 treinada para modelar advecção e dissipação convectiva.
4. **Framework de Benchmark Interinstitucional e Métricas:**
   * Confronto contra baselines (Persistência, Zero-Shot, Lucas-Kanade `pysteps`, LINDA) e fontes oficiais (DECEA/Radar Eusébio, FUNCEME/WRF, INPE/CPTEC, CEMADEN, INMET e ANA).
   * Avaliação por MAE, RMSE, SSIM, PCC, além de métricas de decisão (CSI, FAR, POD, F1-Score) e verificação espacial avançada (**RAPSD**, **SAL** e **FSS**).
5. **Certificação Forense e Rastreabilidade de Hardware:**
   * Execução da Célula Extra de Carimbo Digital para extração de *boot_id* do Linux, GPU UUIDs e geração de *hashes* criptográficos **SHA-256** de todos os artefatos.

---

## Diretrizes de Uso

* Para executar o notebook, recomenda-se a alocação de instâncias com aceleradores de alto desempenho no Google Colab Pro+ com suporte a `BF16`.
* Os resultados numéricos consolidados e os painéis cartográficos exportados são gravados automaticamente nos diretórios [`../results/tables/`](../results/tables/) e [`../results/figures/`](../results/figures/).
* Os relatórios de auditoria e proveniência criptográfica ficam disponíveis em [`../docs/provenance/`](../docs/provenance/).
