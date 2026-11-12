# Arquitetura de Código-Fonte Python (`/src`)

Este diretório contém a base de código modular e refatorada do projeto **F-PluvIA Metropolitan**, desenvolvido como Trabalho de Conclusão de Curso (TCC) no formato de **Artigo Científico** para o Bacharelado em Ciência da Computação do Instituto Federal de Educação, Ciência e Tecnologia do Ceará (IFCE), Campus Maracanaú (semestre letivo `2026.2`).

A arquitetura do código foi projetada seguindo princípios de Engenharia de Software (alta coesão, baixo acoplamento e separação de responsabilidades), permitindo que rotinas complexas de adaptação de domínio, cálculo de métricas espaciais e manipulação geoespacial sejam isoladas do fluxo principal do notebook experimental.

---

## Estrutura de Subpacotes e Módulos

O código-fonte está organizado em três subdiretórios principais, cada um possuindo sua própria documentação detalhada:

```text
src/
├── adapters/          # Classes de compatibilização estrutural e adaptação de domínio
├── metrics/           # Motores de cálculo estatístico e verificação meteorológica
└── utils/             # Funções utilitárias de física de radar, I/O e cartografia
```

### 1. `adapters/` (Design Pattern: GoF Adapter)
* **Finalidade:** Encapsula a lógica de conversão e transformação estrutural entre formatos e domínios incompatíveis sem alterar o código original dos modelos de Machine Learning.
* **Componentes Principais:**
  * `IMERGToMiaradAdapter`: Converte a resolução espacial, recorta as coordenadas da Região Metropolitana de Fortaleza (RMF) e adapta a faixa dinâmica dos dados GPM IMERG-FR v7 para o formato tensorial latente exigido pelo codificador VQGAN do modelo GPTCast.
  * `PyStepsAdapter`: Compatibiliza as matrizes de precipitação para ingestão direta nos algoritmos Lagrangianos de fluxo óptico (Lucas-Kanade) da biblioteca estocástica `pysteps`.
* **Documentação dedicada:** [Consulte `adapters/README.md`](./adapters/README.md)

### 2. `metrics/` (Verificação Meteorológica Multiescala)
* **Finalidade:** Centraliza os algoritmos e funções de avaliação estatística, permitindo o benchmark justo e reprodutível contra modelos institucionais (FUNCEME, CEMADEN, INMET, etc.).
* **Componentes Principais:**
  * **Métricas Contínuas:** Erro Médio Absoluto (MAE), Raiz do Erro Quadrático Médio (RMSE), Índice de Similaridade Estrutural (SSIM) e Coeficiente de Correlação de Pearson (PCC).
  * **Métricas Categóricas (Contingência):** Cálculo matricial para limiares de alerta de Defesa Civil ($1{,}0$, $5{,}0$, $10{,}0$ e $20{,}0\text{ mm/h}$), extraindo CSI, POD, FAR, F1-Score, HSS e ETS.
  * **Métricas Espaciais e Espectrais Avançadas:** 
    * **FSS (*Fractions Skill Score*):** Avaliação multiescala de $2\text{ km}$ a $34\text{ km}$.
    * **SAL (*Structure, Amplitude, Location*):** Decomposição tridimensional de erros em objetos convectivos.
    * **RAPSD:** Densidade Espectral de Potência de Média Radial (Nyquist a $4\text{ km}$).
* **Documentação dedicada:** [Consulte `metrics/README.md`](./metrics/README.md)

### 3. `utils/` (Utilitários Geoespaciais e Físicos)
* **Finalidade:** Agrupa funções auxiliares e matemáticas transversais ao projeto.
* **Componentes Principais:**
  * **Física de Radar:** Implementação da relação empírica $Z\text{–}R$ de Marshall-Palmer ($Z = 200 R^{1{,}6}$) para conversão bidirecional entre refletividade equivalente ($\text{dBZ}$) e taxa de precipitação ($\text{mm/h}$).
  * **Cartografia e Plotagem:** Funções baseadas em `GeoPandas` e `Cartopy` para renderização precisa do contorno dos 19 municípios da RMF e exportação dos painéis a $300\text{ dpi}$.
* **Documentação dedicada:** [Consulte `utils/README.md`](./utils/README.md)

---

## Navegação Cruzada no Repositório

* [Voltar ao README Principal (Raiz do Projeto)](../README.md)
* [Documentação dos Notebooks Experimentais (`../notebooks/README.md`)](../notebooks/README.md)
* [Índice de Resultados, Figuras e Tabelas (`../results/README.md`)](../results/README.md)
