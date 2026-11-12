# Módulo de Adaptadores Estruturais (`/src/adapters`)

Este subpacote implementa o padrão de projeto estrutural **Adapter (GoF - *Gang of Four*)**, atuando como uma camada de tradução (*wrapper*) entre as fontes de dados meteorológicos externos (como estimativas de satélite) e os formatos tensoriais estritos exigidos pelas redes neurais e algoritmos estocásticos do projeto **F-PluvIA Metropolitan** (IFCE Maracanaú, `2026.2`).

A adoção deste padrão de engenharia de software garante que o núcleo (*core*) do modelo GPTCast-8x8 e da biblioteca `pysteps` permaneça fechado para modificação, enquanto o sistema fica aberto para a extensão de novas fontes de dados no futuro (como radares em banda S ou X).

---

## Implementações Principais

### 1. `IMERGToMiaradAdapter`
Classe responsável por realizar a adaptação de domínio físico e geométrico do produto orbital **NASA GPM IMERG Final Run v7** para o domínio nativo de treinamento do **GPTCast-8x8** (originalmente treinado nos Alpes Suíços com dados MIARAD).

**Responsabilidades:**
* **Transformação Geométrica:** Recorta o contorno exato da **Região Metropolitana de Fortaleza (RMF)** e aplica interpolação bicúbica, reamostrando a resolução nativa do satélite ($~10\text{ km}$) para a grade métrica de $2\text{ km}$ ($128 \times 128\text{ pixels}$).
* **Compatibilização de Escala:** Ajusta a faixa dinâmica da taxa de precipitação ($\text{mm/h}$) para garantir que a codificação através do *codebook* congelado do **VQGAN** ($1.024\text{ tokens}$ latentes na grade $8 \times 8$) não sofra degradação ou perda de informação convectiva severa.

### 2. `PyStepsAdapter`
Classe focada na preparação de matrizes de precipitação bidimensionais para ingestão na biblioteca de *nowcasting* estocástico e cinemático `pysteps`.

**Responsabilidades:**
* **Formatação de Tensores:** Modela a sequência histórica de *frames* (*timesteps*) e as dimensões espaciais (X, Y) para os padrões esperados pelos rastreadores de movimento da biblioteca.
* **Integração com Fluxo Óptico:** Alimenta de forma transparente o algoritmo Lagrangiano de fluxo óptico de **Lucas-Kanade** e as rotinas de extrapolação associadas ao baseline **LINDA** (*Pulkkinen et al., 2021*).

---

## Integração com o Pipeline

Estes adaptadores são invocados primariamente nos estágios de **Pré-processamento Geoespacial** e **Benchmark Interinstitucional** do notebook unificado de experimentação, isolando a complexidade do tratamento de matrizes NumPy/PyTorch do fluxo principal do caderno de pesquisa.

---

## Navegação Cruzada no Repositório

* [Voltar ao Guia de Código-Fonte (`../README.md`)](../README.md)
* [Módulo de Métricas e Verificação (`../metrics/README.md`)](../metrics/README.md)
* [Módulo de Utilitários e Cartografia (`../utils/README.md`)](../utils/README.md)
