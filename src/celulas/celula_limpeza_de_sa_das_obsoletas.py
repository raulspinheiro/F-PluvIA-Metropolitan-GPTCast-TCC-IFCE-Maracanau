# @title
# Célula (Limpeza de Saídas Obsoletas): arquiva no Drive e remove do repositório público as 38 saídas de versões
# anteriores do projeto (antes do checkpoint atual 20261116T2123Z): avaliação antiga de 30 janelas, comparação com
# órgãos/instituições (célula removida), versões antigas do teste cego de 2026 e diagnósticos de 11/11/2026.
# Nada é apagado do Drive: os arquivos vão para arquivo_versoes_anteriores_<ENV_TAG>, fora da pasta de resultados,
# e deixam de entrar no Deploy, no Backup e no manifesto SHA-256. 
import base64
import datetime
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

ENV_TAG = globals().get('ENV_TAG', 'G4_RTX6000_Blackwell_95GB')
SERIES_CACHE_DIR = globals().get('SERIES_CACHE_DIR', '/content/drive/MyDrive/GPTCast_RMF/series_2019_2025_2km')
RESULTS_ENV_DIR = Path(globals().get('RESULTS_ENV_DIR', os.path.join(SERIES_CACHE_DIR, f'resultados_{ENV_TAG}')))

# ------------------------------------------------------------------- 1. configuração
SO_LISTAR = False            # 1ª execução: só mostra o que seria feito. Confira e rode de novo com False.
LIMPAR_GITHUB = True        # também remove os mesmos arquivos do repositório público (um commit)
REPO_URL = 'https://github.com/raulspinheiro/F-PluvIA-Metropolitan-GPTCast-TCC-IFCE-Maracanau.git'
GIT_USER_NAME = 'raulspinheiro'
GIT_USER_EMAIL = 'raulspinheiro@users.noreply.github.com'
ARQUIVO_DIR = Path(SERIES_CACHE_DIR) / f'arquivo_versoes_anteriores_{ENV_TAG}'
CORTE_UTC = datetime.datetime(2026, 11, 16, 21, 23)   # treino do checkpoint ativo; arquivo mais novo que isso não é tocado

OBSOLETOS = [n.format(ENV=ENV_TAG) for n in (
    # avaliação antiga com 30 janelas (substituída pela Célula 12 com 1.729 janelas)
    'benchmark_agregado_teste2025_imerg_fr_2km_{ENV}_N30.csv', 'velocidade_inferencia_teste2025_{ENV}_N30.csv',
    'fpluvia_metricas_ml_teste2025_2km_{ENV}_N30.csv', 'fpluvia_resumo_medio_ml_teste2025_{ENV}_N30.csv',
    'painel_agregado_teste2025_2km_{ENV}_N30.png',
    # comparação com órgãos/instituições (célula removida do projeto; números não verificados)
    'comparativo_interinstitucional_leadtime_teste2025_{ENV}_N30.csv',
    'quadro_sintese_interinstitucional_tcc_teste2025_{ENV}_N30.csv',
    'painel_interinstitucional_fpluvia_6orgaos_teste2025_{ENV}_N30.png',
    'tabela_benchmark_6_orgaos_teste_2025_real_{ENV}.csv',
    'metricas_classificacao_benchmark_6_orgaos_teste_2025_{ENV}.png',
    'teste_cego_2026_04_metricas_classificacao_6_orgaos_{ENV}.png',
    # versões antigas do teste cego de 2026 (antes da Célula 13.1 reformulada)
    'tabela_teste_cego_operacional_imerg_rt_2026_{ENV}.csv',
    'teste_cego_2026_01_estratiforme_{ENV}.png', 'teste_cego_2026_01_estratiforme_late_run_20260516_17_{ENV}.png',
    'teste_cego_2026_02_convectivo_{ENV}.png', 'teste_cego_2026_02_convectivo_late_run_20260910_{ENV}.png',
    'teste_cego_2026_03_corrente_{ENV}.png',
    *[f'teste_cego_2026_03_dia_corrente_early_run_{t}_{{ENV}}.png' for t in (
        '20261112_1632', '20261112_1942', '20261112_2033', '20261112_2249', '20261113_0116', '20261113_0130',
        '20261113_0140', '20261113_0153', '20261113_0315', '20261114_0952', '20261114_1005', '20261114_1125',
        '20261114_1127', '20261114_2321', '20261115_1406')],
    # estudo de caso e diagnósticos de versões anteriores (substituídos pelas Células 5 a 11 atuais)
    'fpluvia_metricas_ml_estudo_caso_2km_{ENV}.csv', 'painel_fpluvia_acuracia_precisao_f1_{ENV}.png',
    'diagnostico_vqgan_estagio1_{ENV}.csv', 'painel_vqgan_reconstrucao_{ENV}.png',
    'evolucao_temporal_estudo_caso_{ENV}.csv', 'painel_evolucao_temporal_{ENV}.png',
)]
assert len(OBSOLETOS) == len(set(OBSOLETOS)) == 38, 'lista de obsoletos alterada'
assert not any('_amostragem2024' in n for n in OBSOLETOS), 'a lista não pode conter resultados da execução final'
ALVO = set(OBSOLETOS)
print(f"[Limpeza] Modo: {'SÓ LISTAR (nada será alterado)' if SO_LISTAR else 'EXECUTAR'} | ENV_TAG = {ENV_TAG}")

# ------------------------------------------------- 2. Drive: localizar e arquivar
achados, pulados = [], []
if RESULTS_ENV_DIR.is_dir():
  for p in sorted(RESULTS_ENV_DIR.rglob('*')):
    if p.is_file() and p.name in ALVO:
      mod = datetime.datetime.fromtimestamp(p.stat().st_mtime, datetime.timezone.utc).replace(tzinfo=None)
      (achados if mod < CORTE_UTC else pulados).append((p, mod))
else:
  print(f'[Aviso] Pasta de resultados não encontrada: {RESULTS_ENV_DIR} (rode a Célula 1 e confira o ENV_TAG).')
print(f'[1/3] Drive: {len(achados)} de {len(OBSOLETOS)} obsoletos encontrados em {RESULTS_ENV_DIR.name}'
      + (f'; {len(pulados)} com data posterior ao checkpoint atual, mantidos' if pulados else '') + '.')
for p, mod in pulados:
  print(f'      mantido (mais novo que o corte): {p.relative_to(RESULTS_ENV_DIR)} | {mod:%d/%m/%Y %H:%M} UTC')

registro = []
if achados and not SO_LISTAR:
  ARQUIVO_DIR.mkdir(parents=True, exist_ok=True)
  for p, mod in achados:
    rel = p.relative_to(RESULTS_ENV_DIR)
    dst = ARQUIVO_DIR / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(p), str(dst))
    registro.append({'arquivo': rel.as_posix(), 'data_original_utc': f'{mod:%Y-%m-%d %H:%M}', 'movido_para': str(dst)})
  agora = datetime.datetime.now(datetime.timezone.utc)
  pd.DataFrame(registro).to_csv(ARQUIVO_DIR / f'registro_arquivamento_{agora:%Y%m%dT%H%MZ}.csv', index=False)
  (ARQUIVO_DIR / 'LEIA-ME.txt').write_text(
      'Saídas de versões anteriores do F-PluvIA Metropolitan, geradas antes do checkpoint atual (20261116T2123Z)\n'
      'ou por células removidas/reformuladas. Guardadas só como histórico: não usar no artigo nem na apresentação.\n'
      'Os resultados válidos estão em ' + str(RESULTS_ENV_DIR) + '.\n', encoding='utf-8')
  print(f'      {len(registro)} arquivos movidos para {ARQUIVO_DIR} (registro CSV e LEIA-ME gravados).')
elif achados:
  for p, mod in achados[:6]:
    print(f'      seria movido: {p.relative_to(RESULTS_ENV_DIR)} | {mod:%d/%m/%Y %H:%M} UTC')
  if len(achados) > 6:
    print(f'      ... e mais {len(achados) - 6}.')

# ------------------------------------------- 3. GitHub: remover os mesmos arquivos
if LIMPAR_GITHUB:
  try:
    from google.colab import userdata
    TOKEN = (userdata.get('GITHUB_TOKEN') or '').strip()
  except Exception:
    TOKEN = os.environ.get('GITHUB_TOKEN', '').strip()
  if not TOKEN and REPO_URL.startswith('https://'):
    raise RuntimeError("Defina 'GITHUB_TOKEN' nos Segredos do Colab (ícone de chave) com 'Notebook access' ativado.")
  AUTH = (['-c', 'http.https://github.com/.extraheader=AUTHORIZATION: basic '
           + base64.b64encode(f'x-access-token:{TOKEN}'.encode()).decode()] if TOKEN else [])
  CLONE = Path('/content/github_limpeza' if os.path.isdir('/content') else '/tmp/github_limpeza')

  def git(*args, cwd=CLONE, auth=False):
    r = subprocess.run(['git', *(AUTH if auth else []), *args], cwd=cwd, text=True, capture_output=True)
    if r.returncode:
      msg = (r.stderr + r.stdout).replace(TOKEN, '***') if TOKEN else (r.stderr + r.stdout)
      raise RuntimeError(f'git {args[0]} falhou:\n{msg.strip()}')
    return r.stdout.strip()

  shutil.rmtree(CLONE, ignore_errors=True)
  git('clone', '--depth', '1', REPO_URL, str(CLONE), cwd=None, auth=True)
  base = f'results/{ENV_TAG}/'
  alvo_repo = ALVO - {p.name for p, _ in pulados}       # o que ficou no Drive também fica no repositório
  no_repo = [f for f in git('ls-files').splitlines() if f.startswith(base) and Path(f).name in alvo_repo]
  print(f'[2/3] GitHub: {len(no_repo)} obsoletos encontrados em {base}.')
  if no_repo and not SO_LISTAR:
    git('config', 'user.name', GIT_USER_NAME)
    git('config', 'user.email', GIT_USER_EMAIL)
    git('rm', '-q', '--', *no_repo)
    git('commit', '-q', '-m', 'chore(results): remove saídas obsoletas de versões anteriores',
        '-m', f'{len(no_repo)} arquivos gerados antes do checkpoint atual (20261116T2123Z) ou por células removidas: '
              'avaliação de 30 janelas, comparação com órgãos (não verificada) e versões antigas do teste cego 2026. '
              'Os originais ficam arquivados fora da pasta de resultados.')
    git('push', 'origin', 'HEAD', auth=True)
    print(f'      Commit enviado: {git("rev-parse", "HEAD")[:12]} ({len(no_repo)} arquivos removidos do repositório).')
  elif no_repo:
    for f in no_repo[:6]:
      print(f'      seria removido: {f}')
    if len(no_repo) > 6:
      print(f'      ... e mais {len(no_repo) - 6}.')
  shutil.rmtree(CLONE, ignore_errors=True)
else:
  print('[2/3] GitHub: não alterado (LIMPAR_GITHUB = False).')

print('[3/3] ' + ('Nada foi alterado. Se a lista estiver certa, mude SO_LISTAR para False e rode de novo.' if SO_LISTAR
                  else 'Concluído. Rode depois a Célula Extra 2, a Pseudo-vista, o Deploy e o Backup.'))
