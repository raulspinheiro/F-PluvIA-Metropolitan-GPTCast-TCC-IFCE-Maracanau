# Resultados Experimentais e Artefatos do Benchmark (`/results`)

Este diretório consolida todas as saídas quantitativas, visuais e cartográficas geradas pelo pipeline experimental do projeto **F-PluvIA Metropolitan** (IFCE Maracanaú, `2026.2`). Todos os arquivos presentes aqui são gerados automaticamente pelo *notebook* principal executado no ambiente de aceleração de hardware (Google Colab Pro+).

---

## 1. Escopo da Avaliação

Os resultados armazenados neste diretório refletem a etapa final de inferência e validação do modelo **GPTCast-8x8** (*Fine-Tuned* para a RMF), submetido ao benchmark cruzado (*Interinstitucional e Multiescala*) contra:
* **Baselines Algorítmicos:** Persistência Euleriana, GPTCast-8x8 *Zero-Shot*, Fluxo Óptico de Lucas-Kanade (`pysteps`) e modelo estocástico LINDA.
* **Períodos de Teste:**
  * **Teste Cego (2025):** 4 meses da quadra chuvosa, contabilizando 1.729 sequências temporais fora da amostra de treinamento.
  * **Estudo de Caso Extremo (18/03/2024):** Evento severo chuvoso validado com limiares rígidos da Defesa Civil.
* **Horizontes Temporais (*Lead Times*):** Avaliação preditiva para os próximos $+30$, $+60$, $+90$ e $+120\text{ minutos}$.

---

## 2. Estrutura do Diretório

Os artefatos de saída estão estritamente categorizados em dois subdiretórios principais:

```text
results/
├── figures/           # Imagens, mapas e espectros em alta resolução para o artigo LaTeX
└── tables/            # Planilhas numéricas brutas e tabelas de contingência
```

### [2.1 Catálogo de Figuras a 300 dpi (`/figures`)](./figures/README.md)
Destinado a todos os painéis visuais. Contém:
* Mapas lado a lado de precipitação prevista *vs.* observada (IMERG-FR v7) com o contorno da Região Metropolitana de Fortaleza (Cartopy).
* Gráficos do **FSS (*Fractions Skill Score*)** plotados em curva decaente em função das escalas espaciais ($2\text{ km}$ a $34\text{ km}$).
* Diagramas tridimensionais de fase **SAL (*Structure, Amplitude, Location*)**.
* Espectros de potência de média radial (**RAPSD**).

### [2.2 Tabelas Consolidadas (`/tables`)](./tables/README.md)
Destinado aos dados em formato de planilha (`.csv`). Contém:
* Registros de métricas contínuas (MAE, RMSE, SSIM, PCC) sumarizadas por horizonte de tempo e por modelo.
* Tabelas de contingência e desempenho categórico de alerta (CSI, POD, FAR, F1-Score, HSS, ETS) segregadas pelos limiares críticos de precipitação ($1{,}0$, $5{,}0$, $10{,}0$ e $20{,}0\text{ mm/h}$).

---

## 3. Integridade e Auditoria Forense

**Atenção Avaliadores e Pesquisadores:** Todos os arquivos de imagem (`.png`/`.pdf`) e tabelas numéricas (`.csv`) armazenados nestes subdiretórios possuem sua integridade garantida criptograficamente. O *hash* **SHA-256** individual de cada artefato pode ser consultado nos manifestos forenses emitidos pelo *Carimbo Digital*, disponíveis no diretório [`../docs/provenance/`](../docs/provenance/).

---

## Navegação Cruzada no Repositório

* [Voltar ao README Principal (Raiz do Projeto)](../README.md)
* [Documentação dos Notebooks Experimentais (`../notebooks/README.md`)](../notebooks/README.md)
* [Arquitetura de Código-Fonte Python (`../src/README.md`)](../src/README.md)
