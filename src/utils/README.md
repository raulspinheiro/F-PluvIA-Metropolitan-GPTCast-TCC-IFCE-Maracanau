# Módulo de Utilitários Geoespaciais e Físicos (`/src/utils`)

Este subpacote concentra as funções auxiliares e matemáticas transversais que dão suporte ao pré-processamento de dados e à visualização cartográfica no projeto **F-PluvIA Metropolitan** (IFCE Maracanaú, `2026.2`). 

O isolamento dessas funções no diretório `utils/` evita a repetição de código (*DRY - Don't Repeat Yourself*) no caderno principal e garante que parâmetros físicos constantes (como a relação de Marshall-Palmer) sejam definidos em uma única fonte de verdade.

---

## 1. Conversões Físicas (Radar e Precipitação)

Os modelos de *nowcasting* espacial baseados em arquiteturas de vídeo (como o VQGAN) e bibliotecas de fluxo óptico (como `pysteps`) muitas vezes operam nativamente em domínios de **Refletividade Equivalente ($Z$)** ou **$\text{dBZ}$**, exigindo a conversão precisa de e para **Taxa de Precipitação ($R$, em $\text{mm/h}$)**.

* **Relação $Z\text{–}R$ de Marshall-Palmer:**
  Utilizamos a formulação clássica $Z = a R^b$, com as constantes empíricas padrão de Marshall-Palmer ($a = 200$, $b = 1{,}6$), adequadas para chuva estratiforme e convectiva geral:
  $$ Z = 200 R^{1{,}6} $$
* **Transformação Logarítmica:**
  As rotinas de utilidade convertem os fatores brutos de refletividade $Z$ ($\text{mm}^6/\text{m}^3$) na escala logarítmica decibel $\text{dBZ}$:
  $$ \text{dBZ} = 10 \log_{10}(Z) $$
* **Funções Suportadas:** Conversões bidirecionais (de $\text{mm/h}$ para $\text{dBZ}$ e de $\text{dBZ}$ para $\text{mm/h}$) seguras contra divisões por zero ou logaritmos de limites negativos.

---

## 2. Cartografia e Geometria Espacial (RMF)

A validação visual e espacial das frentes de chuva exige uma projeção cartográfica correta da **Região Metropolitana de Fortaleza (RMF)**, que abrange $19$ municípios do estado do Ceará.

* **Motor Geoespacial (`GeoPandas` e `Cartopy`):**
  * Leitura e manipulação do *shapefile* oficial do IBGE para os limites municipais da RMF.
  * Mascaramento (*clipping*) do tensor espacial de $128 \times 128\text{ pixels}$ ($2\text{ km}$ de resolução) estritamente sobre a área continental e fronteira costeira.
* **Projeção e Coordenadas:** Garantia de manutenção do Sistema de Referência de Coordenadas (CRS) nativo WGS-84 (EPSG:4326) sobreposto à grade plana das predições.

---

## 3. Utilitários de I/O e Exportação Visual

Para garantir a qualidade de submissão do artigo científico, o módulo fornece rotinas padronizadas de salvamento:

* **Exportação em Alta Resolução:** Funções *wrapper* do `matplotlib.pyplot` que salvam automaticamente mapas de contorno, painéis de evolução temporal (horizontes $+30$ a $+120\text{ min}$) e gráficos de linhas (FSS, RAPSD) em formato `.png` e `.pdf` a **$300\text{ dpi}$**.
* **Gestão de Caminhos:** Resolução dinâmica de caminhos absolutos para salvar artefatos na pasta `results/figures/`, independentemente de onde o *script* ou notebook for executado.

---

## Navegação Cruzada no Repositório

* [Voltar ao Guia de Código-Fonte (`../README.md`)](../README.md)
* [Módulo de Adaptadores (`../adapters/README.md`)](../adapters/README.md)
* [Módulo de Métricas e Verificação (`../metrics/README.md`)](../metrics/README.md)
