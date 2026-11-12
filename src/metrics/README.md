# Módulo de Métricas e Verificação Meteorológica (`/src/metrics`)

Este subpacote centraliza o motor de avaliação estatística e espacial do projeto **F-PluvIA Metropolitan** (IFCE Maracanaú, `2026.2`). Ele foi projetado para assegurar um benchmark rigoroso, reprodutível e matematicamente alinhado com as diretrizes da Organização Meteorológica Mundial (OMM) para a avaliação quantitativa de precipitação (QPE) e previsão imediata (*nowcasting*).

Ao separar as métricas do caderno principal (`notebooks/`), o sistema permite testes isolados e padroniza a comparação do modelo **GPTCast-8x8** contra os *baselines* estocásticos (LINDA) e fontes oficiais (FUNCEME, CEMADEN, etc.).

---

## 1. Métricas Contínuas (Fidelidade Pixel a Pixel)

Estas funções avaliam o erro absoluto e a correlação global entre a previsão e a estimativa de satélite (IMERG-FR v7) sobre todo o domínio da Região Metropolitana de Fortaleza:

* **MAE (*Mean Absolute Error*):** Avalia a magnitude média do erro em $\text{mm/h}$.
* **RMSE (*Root Mean Square Error*):** Penaliza grandes discrepâncias locais (comum em picos convectivos).
* **SSIM (*Structural Similarity Index*):** Afere a integridade perceptiva e estrutural das frentes de chuva.
* **PCC (*Pearson Correlation Coefficient*):** Mede a correlação linear das distribuições de intensidade.

## 2. Verificação Categórica (Matriz de Contingência e Alertas)

Essenciais para a Defesa Civil, estas métricas transformam a taxa contínua de chuva em classes dicotômicas (chuva / sem chuva) com base em limiares críticos: **$1{,}0\text{ mm/h}$** (chuva fraca), **$5{,}0\text{ mm/h}$** (moderada), **$10{,}0\text{ mm/h}$** (forte) e **$20{,}0\text{ mm/h}$** (severa).

Baseado em *Acertos (Hits)*, *Falsos Alarmes (FA)* e *Omissões (Misses)*, o módulo extrai:
* **CSI (*Critical Success Index*):** Fração de acertos sobre os eventos totais (desconsiderando os "não-eventos" corretos).
* **POD (*Probability of Detection*):** Taxa de acertos em relação ao total de eventos observados.
* **FAR (*False Alarm Ratio*):** Fração de alarmes disparados sem ocorrência de chuva observada.
* **HSS (*Heidke Skill Score*) & ETS (*Equitable Threat Score*):** Habilidade do modelo corrigida para o grau de acerto que ocorreria aleatoriamente.
* **F1-Score:** Média harmônica entre *Precision* e *Recall* (POD).

## 3. Métricas Espaciais e Espectrais Avançadas

A penalidade de "duplo erro" (*double penalty*) comum em métricas pixel a pixel (como o RMSE) é mitigada neste módulo através da implementação de funções de verificação espacial:

* **FSS (*Fractions Skill Score* - Roberts & Lean, 2008):** Avalia a destreza do modelo ao relaxar a exigência de acerto exato, comparando frações de cobertura de chuva dentro de janelas espaciais crescentes (de $2\text{ km}$ a $34\text{ km}$ no domínio da RMF).
* **SAL (*Structure, Amplitude, Location* - Wernli et al., 2008):** Decompõe o erro em três dimensões objetivas:
  * *Structure (S):* Forma e tamanho dos núcleos convectivos (valores entre -2 e 2).
  * *Amplitude (A):* Volume total de precipitação no domínio (valores entre -2 e 2).
  * *Location (L):* Deslocamento do centro de massa do sistema chuvoso (valores entre 0 e 2).
* **RAPSD (*Radially Averaged Power Spectral Density*):** Analisa a distribuição de energia (variância) no domínio das frequências espaciais, avaliando se o modelo GPTCast-8x8 sofre borrões excessivos nas altas frequências (comprimento de onda de Nyquist em $4\text{ km}$).

---

## Navegação Cruzada no Repositório

* [Voltar ao Guia de Código-Fonte (`../README.md`)](../README.md)
* [Módulo de Adaptadores (`../adapters/README.md`)](../adapters/README.md)
* [Módulo de Utilitários e Cartografia (`../utils/README.md`)](../utils/README.md)
