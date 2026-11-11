# F-PluvIA Metropolitan: Nowcasting de Precipitação na Região Metropolitana de Fortaleza com Modelo GPTCast-8x8 e Benchmark Interinstitucional

Repositório oficial de código, experimentos e artefatos de dados do **Trabalho de Conclusão de Curso (TCC)** — desenvolvido no formato de **artigo científico** — vinculado ao Bacharelado em Ciência da Computação do Instituto Federal de Educação, Ciência e Tecnologia do Ceará (IFCE), Campus Maracanaú, sob o título: *"F-PluvIA Metropolitan: Previsão Imediata de Precipitação Severa na Região Metropolitana de Fortaleza Utilizando o Modelo Autorregressivo GPTCast-8x8 e Validação Cruzada Interinstitucional"*.

---

## Identificação Institucional e Acadêmica

* **Instituição de Ensino:** Instituto Federal de Educação, Ciência e Tecnologia do Ceará (IFCE)
* **Campus:** Campus Maracanaú
* **Curso:** Bacharelado em Ciência da Computação
* **Formato do Trabalho:** Trabalho de Conclusão de Curso (TCC) / Artigo Científico
* **Semestre Letivo:** 2026.2
* **Autor:** Raul Santiago Pinheiro — [Currículo Lattes](https://lattes.cnpq.br/8378024959016497)
* **Orientador:** Prof. Dr. [Amauri Holanda de Souza Júnior] — [Currículo Lattes](http://lattes.cnpq.br/2629504708221802)
* **Área de Concentração:** Inteligência Artificial Aplicada / Visão Computacional / Recursos Hídricos

---

## Visão Geral do Projeto

O **F-PluvIA Metropolitan** investiga o desempenho da inteligência artificial generativa aplicada à previsão meteorológica de curtíssimo prazo (*nowcasting*, horizontes de $+30$, $+60$, $+90$ e $+120\text{ min}$) sobre os 19 municípios da Região Metropolitana de Fortaleza (RMF). O núcleo do sistema baseia-se no ajuste fino (*fine-tuning*) do modelo **GPTCast-8x8**, uma arquitetura híbrida de dois estágios composta por:

1. **Autoencoder Espacial (VQGAN — 90,71 M parâmetros):** Codificador convolucional e livro de códigos discreto (*codebook* de 1.024 tokens) que reduz cada quadro de refletividade/chuva ($128 \times 128\text{ pixels}$) para uma grade latente compacta de $8 \times 8\text{ tokens}$ (onde cada token quantiza uma área física de $16 \times 16\text{ km}$). Mantido congelado durante a adaptação local.
2. **Transformer Temporal (GPT-2 — 304,93 M parâmetros):** Rede autorregressiva com atenção causal treinada para modelar a advecção, crescimento e dissipação das células convectivas tropicais ao longo de 512 tokens espaciotemporais ($8\text{ quadros} \times 64\text{ tokens}$).

O projeto confronta o modelo ajustado contra baselines eulerianos e lagrangianos clássicos, bem como contra a modelagem física e redes de sensores das principais instituições meteorológicas e de monitoramento do Brasil e do Ceará.

---

## Destaques Metodológicos e Estrutura de Dados

* **Série Plurianual e Domínio Cartográfico:** 28 meses de dados históricos de satélite em alta resolução (**GPM IMERG-FR v7**), interpolados bicubicamente para uma grade métrica contínua de **$2\text{ km}$ ($128 \times 128\text{ pixels}$ / $256 \times 256\text{ km}$)** com pré-alocação contígua em RAM (*Zero-Copy*).
* **Particionamento Temporal Estrito (Anti-Data Leakage):**
  * **Treinamento (2019–2023):** 20 meses de quadras chuvosas locais (fevereiro a maio) utilizados no ajuste do Transformer em precisão mista `BF16`.
  * **Validação (2024):** 4 meses completos (incluindo o estudo de caso severo de 18/03/2024) com critério de parada antecipada (*Early Stopping*) ancorado na perda latente (`val/loss = 4.822`).
  * **Teste Científico Fora da Amostra (2025):** 4 meses independentes do IMERG-FR (1.729 sequências temporais) para verificação estatística cega.
* **Avaliação Multidimensional:**
  * **Métricas Físicas e Contínuas:** MAE, RMSE, SSIM e Correlação de Pearson (PCC).
  * **Métricas de Classificação e Decisão (F-PluvIA):** Acurácia Global, Precisão (PPV), Recall/Sensibilidade (POD), F1-Score, Taxa de Falso Alarme (FAR) e Índice de Sucesso Crítico (CSI / IoU) avaliados nos limiares operacionais de $0{,}5\text{ mm/h}$ (fraca), $1{,}0\text{ mm/h}$ (moderada) e $5{,}0\text{ mm/h}$ (severa).
  * **Verificação Espacial e Espectral Avançada:** Densidade Espectral de Potência Radialmente Mediada (**RAPSD**) na escala de Nyquist ($4\text{ km}$), Decomposição Tridimensional do Erro (**Diagrama SAL** — Estrutura, Amplitude, Localização e deslocamento do baricentro em km) e **Fractions Skill Score (FSS)** avaliado em escala municipal multirresolução ($2\text{ km}$ a $34\text{ km}$, com foco na transição pós-token em $18\text{ km}$).
* **Framework de Benchmark Interinstitucional:**
  * **Modelos Físicos e Baselines:** Persistência Euleriana, GPTCast-8x8 Zero-Shot, Fluxo Óptico de Lucas-Kanade (`pysteps`) e LINDA (*Lagrangian Integro-Difference Autoregression* — Pulkkinen et al., 2021).
  * **Fontes Institucionais Confrontadas:** DECEA (Radar Doppler Banda-S Eusébio/REDEMET), FUNCEME (WRF Regional 5 km), INPE/CPTEC (Hidroestimador GOES-16 ABI IR / BAM), CEMADEN (Rede de PCDs pluviométricas urbanas na RMF), INMET (Estação Sinótica A305 Fortaleza + NWP) e ANA (Telemetria Hidrológica de Bacias).
* **Rastreabilidade e Hardware:** Pipeline instrumentado para quantificar *speedup*, eficiência de VRAM e consumo de Unidades de Computação (CUs) no Google Colab Pro+ comparando as arquiteturas **NVIDIA RTX 6000 Blackwell (G4 — 95 GB GDDR7)** e **NVIDIA H100 Hopper (80 GB HBM3)**, acompanhado de geração automática de certificados de auditoria digital com hash SHA-256 dos artefatos.

---

## Organização do Repositório

![Organização do Repositório](docs/estrutura_repositorio.jpg)

```text
├── notebooks/
│   └── F_PluvIA_Metropolitan_GPTCast8_Benchmark.ipynb   # Pipeline experimental unificado (Células 1 a 13.1)
├── src/
│   ├── adapters/          # Implementações do GoF Adapter (IMERGToMiaradAdapter, PyStepsAdapter)
│   ├── metrics/           # Módulos de cálculo de SAL, RAPSD, FSS e métricas de contingência ML
│   └── utils/             # Funções de conversão Z-R (Marshall-Palmer) e cartografia GeoPandas/Cartopy
├── results/
│   ├── figures/           # Painéis cartográficos a 300 dpi (Mapas RMF, RAPSD, SAL, Curvas FSS)
│   └── tables/            # Relatórios consolidados em CSV por horizonte (+30 a +120 min) e ambiente
└── docs/
    ├── estrutura_repositorio.jpg   # Diagrama visual dos artefatos do repositório
    ├── references.bib              # Base de referências bibliográficas em formato BibTeX
    └── provenance/                 # Certificados de auditoria forense da execução em GPU e hashes SHA-256
