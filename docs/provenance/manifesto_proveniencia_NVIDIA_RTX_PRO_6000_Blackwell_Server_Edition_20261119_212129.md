# Manifesto de integridade e proveniência — F-PluvIA Metropolitan

**Emissão:** 19/11/2026 21:21:29 (Fortaleza) · 2026-11-20 00:21:29 UTC  
**Ambiente:** NVIDIA RTX PRO 6000 Blackwell Server Edition (95.0 GB) · GPU NVIDIA RTX PRO 6000 Blackwell Server Edition · Python 3.13.16  
**Autor:** Raul Santiago Pinheiro · **Orientador:** Amauri Holanda de Souza Júnior · Instituto Federal de Educação, Ciência e Tecnologia do Ceará (IFCE) — Campus Maracanaú · 2026.2

## O que este manifesto comprova

Que os arquivos listados não mudaram desde a emissão: basta recalcular o SHA-256 de cada um (ou rodar "sha256sum -c" no arquivo SHA256SUMS). Não comprova autoria nem o hardware usado; esses campos são declarados pelo ambiente no momento da emissão.

**Hash raiz (SHA-256 da lista de arquivos):** `c062ebabce521ff5aac739fdd3a1bce8a26b9108a1d78e2ba34fb3360a4a9baa`

## Artefatos por pasta

| Pasta | Arquivos | Tamanho |
|---|---:|---:|
| checkpoints | 7 | 19,301.5 MB |
| comparativo_hardware | 3 | 0.2 MB |
| diagramas_uml | 70 | 5.6 MB |
| notebook | 1 | 9.5 MB |
| resultados | 472 | 3,133.6 MB |
| series_2km | 36 | 1,330.1 MB |

## Execução do modelo

- **checkpoint:** `gptcast8_rmf_2km_G4_RTX6000_Blackwell_95GB_20261116T2123Z_best-00-4.7167.ckpt`
- **digest_transformer_gpt8:** `2a17da25e1d8`

## Bibliotecas

- gpm-api 0.4.0
- pysteps 1.21.5
- torch 2.14.1+cu130
- lightning 2.6.6
- pytorch-lightning 2.6.6
- numpy 2.4.6
- pandas 2.2.3
- xarray 2025.12.0
- scikit-image 0.25.2
- cartopy 0.26.0
- geopandas 1.2.0

## Código das células (SHA-256 do código executado)

| Célula | Linhas | SHA-256 |
|---|---:|---|
| Célula (Machine id) | 23 | `e23b39c7329a8aae…` |
| Célula (Security info) | 38 | `95d1ef372ae138e1…` |
| Célula 1 | 160 | `25eb45ed6a58258d…` |
| Célula (Limpeza de Saídas Obsoletas) | 143 | `0690e565d2459945…` |
| Célula 2 | 426 | `395d509c894286b8…` |
| Célula 3 | 373 | `fa150491f5dbb360…` |
| Célula 4 | 401 | `06b3e30690a908fc…` |
| Célula 4 (conferência rápida) | 6 | `e87d93247e5d9a6b…` |
| Célula 4.1 | 241 | `284d91c2d62b0e96…` |
| Célula 4.2 | 152 | `2788e9d1cb292486…` |
| Célula 4.3 (Opcional - Fine-Tuning Suave - Adds Another Dimension of Search) | 346 | `90db637d3177bb69…` |
| Célula 4.4 | 263 | `05a2cf599175dfe0…` |
| Célula 4.5 | 340 | `e7950c6d8651d1bd…` |
| Célula (Activates Checkpoint) | 151 | `e68095f8d3ea7907…` |
| Célula 5 | 174 | `88f7888458faac68…` |
| Célula 6 | 153 | `f4d33840367afd03…` |
| Célula 7 | 161 | `f5a4b268449c107d…` |
| Célula 8 | 159 | `569989cc12cc004d…` |
| Célula 9 | 216 | `ea5a5255f8862609…` |
| Célula 10 | 158 | `1a597bd8bbdc39f6…` |
| Célula 11 | 144 | `2846484678410dc1…` |
| Célula 12 | 464 | `07804e5909cb00a2…` |
| Célula 12.1 | 151 | `a2d7069ea581d847…` |
| Célula 13 | 149 | `1d1daf2d4f5c6aaa…` |
| Célula 13.1 | 594 | `380c5fc32eee9633…` |
| Célula 13.2 | 259 | `7f72843222d51a61…` |
| Célula 13.3 | 245 | `77bba557c0289487…` |
| Célula 13.4 | 287 | `8f82f8c9f42eed4a…` |
| Célula Extra 1 | 268 | `680ce0aa797a2f77…` |
| Célula Full UMLs | 397 | `5b475052f170b4ad…` |
| Célula Extra 2 | 278 | `72f6d988fa0d4309…` |
| Célula Base (Pseudo-vista) | 364 | `3688e9a18d711172…` |
| Célula (Deploy Automatizado) | 312 | `773c9ff7609f436b…` |
| Célula Backup | 189 | `b3b7c3ceaf2263ed…` |

## Lista completa

Todos os 589 arquivos e hashes estão em `SHA256SUMS_20261119_212129.txt` e `manifesto_proveniencia_NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition_20261119_212129.json`.
Para conferir: `sha256sum -c SHA256SUMS_20261119_212129.txt` (rodado na pasta que contém resultados/, checkpoints/ e series_2km/).
