# Catálogo de Figuras e Painéis Cartográficos (`/results/figures`)

Este subdiretório atua como o repositório central para todos os artefatos visuais gerados automaticamente durante a avaliação e inferência do modelo **F-PluvIA Metropolitan** (IFCE Maracanaú, `2026.2`). 

Os gráficos e mapas aqui armazenados estão formatados e padronizados para inclusão direta em documentos acadêmicos renderizados via **LaTeX / Overleaf**, obedecendo aos critérios de qualidade de periódicos científicos.

---

## 1. Tipologia dos Artefatos Visuais

### 1.1 Mapas de Precipitação (Cartografia)
* **Descrição:** Painéis comparativos lado a lado (*Ground Truth* IMERG-FR v7 *vs.* Modelos Base *vs.* GPTCast-8x8 *Fine-Tuned*).
* **Características Técnicas:** Renderizados sobrepondo os tensores de predição ao contorno geográfico dos 19 municípios da Região Metropolitana de Fortaleza (RMF) utilizando `Cartopy` e `GeoPandas`.
* **Projeção:** Coordenadas Geográficas Padrão (WGS-84 / EPSG:4326).
* **Escala de Cores:** Mapa de cores (*colormap*) não-linear padronizado para refletividade e precipitação convectiva (destacando picos $> 20\text{ mm/h}$).

### 1.2 Gráficos de Verificação Espacial Multiescala
Em contraste com os mapas puramente visuais, estes diagramas fornecem a representação gráfica do desempenho do modelo segundo as diretrizes de avaliação espacial:
* **Curvas FSS (*Fractions Skill Score*):** Gráficos de linha plotando a destreza do modelo (eixo Y) em função de diferentes escalas de vizinhança (eixo X, de $2\text{ km}$ a $34\text{ km}$), com a linha de referência do limite de habilidade útil (*Uniform Skill*).
* **Diagramas de Fase SAL:** Representação em espaço tridimensional (ou dispersão 2D) ilustrando o desempenho do modelo quanto à **Estrutura (S)**, **Amplitude (A)** e **Localização (L)** dos núcleos chuvosos.
* **Espectro RAPSD:** Gráfico log-log que plota a Densidade Espectral de Potência de Média Radial em relação aos comprimentos de onda de Nyquist, revelando a preservação (ou suavização) das altas frequências espaciais geradas pelo codificador VQGAN.

---

## 2. Formato e Resolução de Exportação

A geração destes artefatos é controlada pelas rotinas utilitárias presentes em [`../../src/utils/`](../../src/utils/).
* **Resolução Obrigatória:** Todos os painéis são salvos com densidade fixa de **300 dpi** (*dots per inch*), garantindo nitidez vetorial e raster adequada para impressão científica.
* **Formatos de Arquivo:**
  * **`.pdf`:** Formato preferencial, gerando gráficos vetoriais nativos que não perdem resolução ao aplicar *zoom* em visualizadores de PDF (ideal para curvas FSS e RAPSD).
  * **`.png`:** Formato raster de alta qualidade com fundo transparente (*alpha channel* opcional), otimizado para os painéis cartográficos densos.

---

## 3. Rastreabilidade Criptográfica

Para assegurar que nenhuma figura foi adulterada ou "embelezada" pós-processamento, todos os arquivos `.png` e `.pdf` depositados nesta pasta têm sua assinatura binária validada. O *hash* **SHA-256** de cada imagem consta obrigatoriamente no certificado de proveniência gerado no Google Colab Pro+, localizado em [`../../docs/provenance/`](../../docs/provenance/).

---

## Navegação Cruzada no Repositório

* [Voltar ao Índice de Resultados (`../README.md`)](../README.md)
* [Dicionário de Dados CSV (`../tables/README.md`)](../tables/README.md)
* [Documentação de Utilitários Cartográficos (`../../src/utils/README.md`)](../../src/utils/README.md)
