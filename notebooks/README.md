# Diretório de Notebooks Experimentais (`/notebooks`)

Este diretório armazena o caderno computacional unificado de execução, treinamento, inferência e auditoria do projeto **F-PluvIA Metropolitan**, desenvolvido como Trabalho de Conclusão de Curso (TCC) no formato de **Artigo Científico** vinculado ao Bacharelado em Ciência da Computação do Instituto Federal de Educação, Ciência e Tecnologia do Ceará (IFCE), Campus Maracanaú (semestre letivo `2026.2`).

---

## 1. Identificação do Artefato Canônico

```text
notebooks/
├── README.md                                                                 # Guia arquitetural e operacional (este arquivo)
└── Model_Annual_Full_Training_Series_TCC2_-_2026_2_-_RAUL_SANTIAGO_PINHEIRO.ipynb # Pipeline experimental completo
```

* **Arquivo Principal:** `Model_Annual_Full_Training_Series_TCC2_-_2026_2_-_RAUL_SANTIAGO_PINHEIRO.ipynb`
* **Preservação de Nomenclatura Canônica:** O nome original do arquivo de implementação no Google Colab Pro+ é estritamente mantido neste diretório para garantir a correspondência binária e criptográfica (`SHA-256`) com os manifestos forenses armazenados em [`../docs/provenance/`](../docs/provenance/).
* **Infraestrutura Homologada:** Google Colab Pro+ com avaliação comparativa em aceleradores **NVIDIA RTX 6000 Blackwell (G4 — 95 GB GDDR7)** e **NVIDIA H100 Hopper (80 GB HBM3)**.

---

## 2. Arquitetura Sequencial do Pipeline (Células 1 a 13.1 + Células Extras)

O notebook foi estruturado de forma monolítica e autocontida para assegurar reprodutibilidade científica ponta a ponta sem dependências ocultas:

### Estágio I: Configuração de Ambiente e Ingestão Geoespacial (GPM IMERG-FR v7)
* Montagem de armazenamento persistente, verificação de telemetria CUDA/`BF16` e ingestão de 28 meses de dados históricos das quadras chuvosas do Ceará (fevereiro a maio, $2019\text{–}2025$).
* Recorte espacial centrado na **Região Metropolitana de Fortaleza (RMF — 19 municípios)** e reamostragem bicúbica para grade métrica regular de $2\text{ km}$ ($128 \times 128\text{ pixels}$, cobrindo $256 \times 256\text{ km}$) com alocação em modo *Zero-Copy*.

### Estágio II: Particionamento Temporal Estrito e Adaptação Física ($Z\text{–}R$)
* Aplicação do padrão de projeto **GoF Adapter** (`IMERGToMiaradAdapter`) utilizando a relação empírica de **Marshall-Palmer** ($Z = 200 R^{1{,}6}$) para compatibilizar a taxa de precipitação orbital ($\text{mm/h}$) com a distribuição radiométrica latente do autoencoder pré-treinado.
* Divisão cronológica sem vazamento de informação (*anti-data leakage*):
  * **Treinamento ($2019\text{–}2023$):** 20 meses para ajuste fino supervisionado.
  * **Validação ($2024$):** 4 meses para monitoramento de perda e *Early Stopping* ($\text{val/loss} = 4{,}822$), incluindo o evento extremo de $18/03/2024$.
  * **Teste Cego ($2025$):** 4 meses ($1.729$ sequências independentes) para avaliação fora da amostra (*out-of-sample*).

### Estágio III: Fine-Tuning Regional do Modelo GPTCast-8x8
* **Tokenizer Espacial VQGAN ($90{,}71\text{ M}$ de parâmetros):** Mantido com pesos congelados (*frozen codebook* de $1.024\text{ tokens}$ discretos sobre grade latente $8 \times 8$) para preservar a morfologia de alta frequência das células convectivas.
* **Transformer Temporal Causal GPT-2 ($304{,}93\text{ M}$ de parâmetros):** Ajustado em precisão mista nativa `BF16` com otimizador AdamW para modelar a dinâmica de advecção costeira, ZCIT, VCAN e linhas de instabilidade nos horizontes de $+30, +60, +90\text{ e }+120\text{ min}$.

### Estágio IV: Benchmark Interinstitucional e Verificação Multiescala (Células Finais até 13.1)
* Comparação sistemática contra *baselines* cinemáticos e estocásticos: Persistência Euleriana, GPTCast-8x8 Zero-Shot, Fluxo Óptico Lagrangiano Lucas-Kanade (`pysteps`) e **LINDA** (*Pulkkinen et al., 2021*), além de referenciamento cruzado com produtos operacionais (DECEA/Radar Eusébio, FUNCEME, CEMADEN, INMET, ANA e INPE/CPTEC).
* Geração automatizada de relatórios `.csv` (MAE, RMSE, SSIM, PCC, CSI, POD, FAR, F1-Score, HSS, ETS) e exportação de painéis cartográficos e espectrais a $300\text{ dpi}$ (**RAPSD** com escala de Nyquist em $4\text{ km}$, diagramas **SAL** e curvas **FSS** de $2\text{ km}$ a $34\text{ km}$).

### Estágio V: Auditoria Criptográfica e Deploy Automatizado (Células Extras 1 e 2)
1. **Célula Extra 1 (Carimbo Digital):** Captura o `boot_id` do kernel Linux, o **GPU UUID** físico da placa NVIDIA, calcula o *hash* **SHA-256** de todos os artefatos gerados (incluindo o próprio `.ipynb`) e emite os certificados em formato duplo (`certificado_autoria_*.json` e `.md`) em [`../docs/provenance/`](../docs/provenance/).
2. **Célula Extra 2 (Sincronização e Upload GitHub):** Autentica via `Colab Secrets` (`GITHUB_TOKEN`), filtra arquivos acima de $95\text{ MB}$ (retendo pesos brutos no Google Drive) e executa `git commit` e `git push` automatizados para este repositório.

---

## 3. Protocolo Operacional de Execução no Google Colab Pro+

1. Execute sequencialmente da **Célula 1 até a Célula 13.1**.
2. Ao concluir a Célula 13.1, pressione **`Ctrl + S`** e aguarde a confirmação de salvamento no Google Drive para consolidar os *outputs* dentro do arquivo `.ipynb`.
3. Execute a **Célula Extra 1 (Carimbo Digital)** para lacrar os *hashes* SHA-256.
4. Execute imediatamente a **Célula Extra 2 (Upload GitHub)** sem realizar novas edições intermediárias, garantindo que o arquivo `.ipynb` enviado ao repositório mantenha exatamente o mesmo *hash* registrado no certificado de proveniência.

---

## Navegação Cruzada no Repositório

* [Voltar ao README Principal (Raiz do Projeto)](../README.md)
* [Arquitetura de Código-Fonte Python (`../src/README.md`)](../src/README.md)
* [Índice de Resultados, Figuras e Tabelas (`../results/README.md`)](../results/README.md)
* [Protocolo de Auditoria e Proveniência (`../docs/provenance/README.md`)](../docs/provenance/README.md)
