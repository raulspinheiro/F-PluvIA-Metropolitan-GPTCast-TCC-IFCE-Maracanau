# Documentação do Diretório /docs

Este diretório armazena os artefatos de suporte administrativo, metodológico, visual, bibliográfico e de auditoria forense do projeto **F-PluvIA Metropolitan**, desenvolvido como Trabalho de Conclusão de Curso (TCC) em formato de **Artigo Científico** vinculado ao Bacharelado em Ciência da Computação do Instituto Federal de Educação, Ciência e Tecnologia do Ceará (IFCE), Campus Maracanaú (semestre letivo `2026.2`).

---

## Estrutura e Descrição dos Artefatos Armazenados

Abaixo está o detalhamento atualizado dos arquivos e subpastas mantidos sob este diretório:

```text
docs/
├── structure.md                # Guia descritivo da estrutura e dos artefatos da pasta /docs (este arquivo)
├── estrutura_repositorio.jpg   # Diagrama visual ilustrativo da arquitetura de diretórios do projeto
├── references.bib              # Base de dados bibliográfica unificada em formato BibTeX (LaTeX/ABNT)
└── provenance/                 # Diretório de auditoria e certificação forense da execução em GPU
    ├── README.md                      # Protocolo de verificação criptográfica SHA-256 e telemetria
    ├── certificado_autoria_*.json     # Manifestos estruturados (SHA-256, boot_id e GPU UUID)
    └── certificado_autoria_*.md       # Relatórios legíveis com tabelas de auditoria renderizáveis no GitHub
```

### 1. `structure.md`
* **Finalidade:** Documento de índice e governança da pasta `/docs`, orientando avaliadores e pesquisadores sobre a localização das referências formais e dos manifestos de reprodutibilidade.

### 2. `estrutura_repositorio.jpg`
* **Finalidade:** Arquivo visual de apoio utilizado para ilustrar a divisão modular do repositório entre código-fonte, cadernos de experimentação, resultados e documentação.

### 3. `references.bib`
* **Finalidade:** Repositório central de referências bibliográficas estruturado em padrão **BibTeX**. Contém as entradas formais das bases teóricas, métricas espaciais e modelos de inteligência artificial adotados no artigo científico, incluindo:
  * **Arquitetura Base e Tokenização Latente:** *Franch et al. (2024)* para o modelo **GPTCast** e *Esser et al. (2021)* para o **VQGAN**;
  * **Baselines Lagrangianos e Estocásticos:** *Pulkkinen et al. (2019)* para a biblioteca `pysteps` e *Pulkkinen et al. (2021)* para o modelo **LINDA**;
  * **Métricas Espaciais e Espectrais:** *Roberts & Lean (2008)* para o **FSS** (*Fractions Skill Score*) e *Wernli et al. (2008)* para a decomposição tridimensional **SAL** (*Structure, Amplitude, Location*);
  * **Física de Radar e Sensoriamento Orbital:** *Marshall & Palmer (1948)* para a relação $Z\text{–}R$ e documentação técnica do produto **NASA GPM IMERG Final Run v7**.

### 4. `provenance/` (Pasta de Rastreabilidade e Auditoria Forense)
* **Finalidade:** Armazena os artefatos de certificação gerados automaticamente ao término do pipeline experimental no **Google Colab Pro+** (Célula Extra — Carimbo Digital).
* **Conteúdo:** Registra em formato duplo (`.json` para validação automatizada e `.md` para inspeção visual no GitHub) os *hashes* criptográficos **SHA-256** de todos os arquivos gerados, o identificador de inicialização do kernel Linux (`boot_id`), os identificadores físicos unívocos (**GPU UUID**) das placas NVIDIA utilizadas (**RTX 6000 Blackwell / G4 — 95 GB GDDR7** vs. **H100 Hopper — 80 GB HBM3**) e a assinatura mestre da sessão (*Master Session Hash*).
* **Documentação Interna:** Consulte [`provenance/README.md`](./provenance/README.md) para instruções de verificação via terminal (`sha256sum` / `Get-FileHash`).

---

## Navegação Cruzada no Repositório

* [README Principal (Raiz do Projeto)](../README.md)
* [Documentação dos Notebooks Experimentais (`../notebooks/README.md`)](../notebooks/README.md)
* [Arquitetura de Código-Fonte Python (`../src/README.md`)](../src/README.md)
* [Índice de Resultados, Figuras e Tabelas (`../results/README.md`)](../results/README.md)
