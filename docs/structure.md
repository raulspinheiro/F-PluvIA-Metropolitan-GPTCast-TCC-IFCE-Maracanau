# Documentação do Diretório /docs

Este diretório armazena os artefatos de suporte administrativo, metodológico, visual e bibliográfico do projeto **F-PluvIA Metropolitan**, desenvolvido como Trabalho de Conclusão de Curso (TCC) em formato de **Artigo Científico** vinculado ao Bacharelado em Ciência da Computação do Instituto Federal de Educação, Ciência e Tecnologia do Ceará (IFCE), Campus Maracanaú (semestre letivo 2026.2).

---

## Estrutura e Descrição dos Artefatos Armazenados

1. estrutura_repositorio.jpg
Finalidade: Arquivo de imagem utilizado no README.md principal da raiz para ilustrar visualmente a arquitetura e a disposição lógica dos diretórios (notebooks/, src/, results/ e docs/).

2. references.bib
Finalidade: Repositório central de referências bibliográficas estruturado em padrão BibTeX. Contém as entradas formais das bases teóricas, métricas espaciais e modelos de inteligência artificial adotados no artigo (incluindo Franch et al., 2024 para o GPTCast, Pulkkinen et al., 2021 para o LINDA, Roberts & Lean, 2008 para o FSS e Wernli et al., 2008 para o SAL).

3. provenance/ (Pasta de Rastreabilidade e Auditoria)
Finalidade: Destinada a armazenar os arquivos em formato JSON gerados automaticamente ao término do pipeline no Google Colab Pro+.

Conteúdo: Registram hashes criptográficos SHA-256 de todos os artefatos produzidos, IDs de inicialização do container Linux (boot_id) e identificadores unívocos das GPUs NVIDIA utilizadas nas execuções comparativas (G4 Blackwell vs. H100 Hopper), assegurando a reprodutibilidade científica e a integridade da autoria.

Diretrizes de Uso e Contribuição
Todos os artefatos textuais e bibliográficos contidos nesta pasta seguem as normas de formatação compatíveis com editores LaTeX (como o Overleaf) e padrões acadêmicos exigidos pelo IFCE.

Para citar o projeto ou utilizar os códigos e bases bibliográficas deste repositório, consulte as instruções detalhadas na seção de citação do README.md principal.

Abaixo está o detalhamento dos arquivos e subpastas mantidos sob este diretório:

```text
docs/
├── estrutura_repositorio.jpg   # Diagrama visual descritivo da árvore de diretórios do projeto
├── references.bib              # Base de dados bibliográfica unificada em formato BibTeX (LaTeX)
└── provenance/                 # Diretório de auditoria e certificação forense da execução em GPU
    └── certificado_autoria_*.json # Carimbos criptográficos SHA-256 e metadados de hardware da sessão

