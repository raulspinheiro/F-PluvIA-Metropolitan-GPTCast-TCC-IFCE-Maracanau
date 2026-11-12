# Dicionário de Dados e Relatórios Tabulares (`/results/tables`)

Este subdiretório armazena as planilhas consolidadas em formato **`.csv` (Comma-Separated Values)** geradas pelo módulo de métricas do projeto **F-PluvIA Metropolitan** (IFCE Maracanaú, `2026.2`). 

Os relatórios aqui contidos oferecem o embasamento numérico bruto para o benchmark interinstitucional, permitindo a extração rápida de tabelas para o artigo final em LaTeX ou reanálises estatísticas.

---

## 1. Escopo das Planilhas de Avaliação

Os arquivos `.csv` estão divididos primariamente por conjunto de dados avaliado:
* **`benchmark_rmf_test2025.csv`:** Agregação da avaliação contínua sobre as 1.729 sequências independentes extraídas da quadra chuvosa de 2025 (Teste Cego).
* **`benchmark_rmf_event_20240318.csv`:** Avaliação focada estritamente nos *frames* do evento de precipitação severa ocorrido em 18 de março de 2024, dissecando a degradação temporal do modelo frente a um sistema convectivo extremo.

---

## 2. Dicionário de Dados (Esquema das Colunas)

Cada arquivo gerado possui uma estrutura tabular padronizada contendo os seguintes campos:

### 2.1 Metadados do Experimento
* `Model`: Nome do modelo avaliado (ex: `GPTCast-8x8_FT`, `LINDA`, `PySteps_LK`, `Persistence`).
* `Lead_Time_Min`: Horizonte preditivo da estimativa em minutos (`+30`, `+60`, `+90`, `+120`).

### 2.2 Métricas Contínuas (Fidelidade Global)
* `MAE`: Erro Médio Absoluto expresso em $\text{mm/h}$. Valores menores indicam melhor desempenho.
* `RMSE`: Raiz do Erro Quadrático Médio ($\text{mm/h}$). Mais sensível a falhas em picos convectivos isolados.
* `SSIM`: Índice de Similaridade Estrutural (0 a 1). Valores próximos a 1 indicam morfologia visual semelhante ao satélite.
* `PCC`: Coeficiente de Correlação de Pearson (-1 a 1).

### 2.3 Métricas Categóricas (Matriz de Contingência)
Calculadas iterativamente para os limiares críticos da Defesa Civil: $1{,}0\text{ mm/h}$, $5{,}0\text{ mm/h}$, $10{,}0\text{ mm/h}$ e $20{,}0\text{ mm/h}$. A nomenclatura das colunas segue o formato `<Metrica>_Thr<Limiar>`, por exemplo: `CSI_Thr10`.
* `CSI`: *Critical Success Index* (0 a 1). Principal métrica de alerta.
* `POD`: *Probability of Detection* (0 a 1). Taxa de acerto real (Recall).
* `FAR`: *False Alarm Ratio* (0 a 1). Proporção de alarmes falsos emitidos. Valores menores são desejáveis.
* `F1`: F1-Score (0 a 1). Média harmônica entre precisão e POD.
* `HSS`: *Heidke Skill Score* (-1 a 1). Habilidade do modelo acima do acerto aleatório.
* `ETS`: *Equitable Threat Score* (-1/3 a 1). Habilidade do modelo corrigida para "não-eventos".

---

## 3. Rastreabilidade Criptográfica

Para fins de auditoria de pesquisa, cada planilha `.csv` depositada neste diretório no momento da execução no Google Colab Pro+ é registrada com um *hash* **SHA-256** irrevogável. Caso você deseje importar esses dados para processamento externo, recomenda-se aferir a integridade da planilha utilizando o manifesto armazenado em [`../../docs/provenance/`](../../docs/provenance/).

---

## Navegação Cruzada no Repositório

* [Voltar ao Índice de Resultados (`../README.md`)](../README.md)
* [Catálogo de Figuras a 300 dpi (`../figures/README.md`)](../figures/README.md)
* [Módulo de Métricas e Verificação (`../../src/metrics/README.md`)](../../src/metrics/README.md)
