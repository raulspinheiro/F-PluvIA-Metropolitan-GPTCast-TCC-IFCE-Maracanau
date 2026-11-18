# @title
# Célula 4.2: Curva de Treino do Fine-Tuning — perda de treino x validação e diferença entre elas, com cabeçalho
# do contexto do treino. Lê o treino registrado como ativo: o da Célula 4 ou, se promovido, o da Célula 4.3
# (validação a cada fração de época).
#
# Requisito de Execução: rodar depois da Célula 4.
#
import json
import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rec = json.load(open(training_record_json, encoding='utf-8'))
m = pd.read_csv(rec['curva_por_epoca_csv'])

col_v = next((c for c in ('val/loss_epoch', 'val/loss') if c in m.columns), None)
col_te = 'train/loss_epoch' if 'train/loss_epoch' in m.columns else None
col_ts = next((c for c in ('train/loss_step', 'train/loss') if c in m.columns), None)
if col_v is None or (col_te is None and col_ts is None):
  raise RuntimeError(f'O metrics.csv não tem as colunas de perda esperadas; encontradas: {list(m.columns)}')

# Validação por época (Célula 4) ou a cada fração de época (Célula 4.3, val_check_interval < 1).
VC = float(rec.get('val_check_interval') or 1.0)
v = m.dropna(subset=[col_v])[['epoch', 'step', col_v]].drop_duplicates('step').sort_values('step').reset_index(drop=True)
FRACAO = VC < 1.0 or v.groupby('epoch').size().max() > 1
if FRACAO:
  v['x'] = v['epoch'] + (v.groupby('epoch').cumcount() + 1) * VC       # 0,25; 0,50; ...; 1,00 = fim da época 1
  if col_ts is not None:
    # Treino no trecho entre uma validação e a anterior (média dos passos desse trecho).
    t = m.dropna(subset=[col_ts])[['step', col_ts]]
    lim = [-1] + v['step'].tolist()
    v['treino'] = [t.loc[(t['step'] > a) & (t['step'] <= b), col_ts].mean() for a, b in zip(lim[:-1], lim[1:])]
  else:
    v['treino'] = v['epoch'].map(m.groupby('epoch')[col_te].mean())
  passo_x = VC
else:
  curva_ep = m.groupby('epoch')[[c for c in (col_te, col_v) if c]].mean().dropna(how='all')
  v = pd.DataFrame({'x': curva_ep.index.astype(int) + 1, col_v: curva_ep[col_v].values,
                    'treino': curva_ep[col_te].values if col_te else np.nan})
  passo_x = 1.0
v = v.dropna(subset=[col_v])
x, va, tr = v['x'].to_numpy(float), v[col_v].to_numpy(float), v['treino'].to_numpy(float)
gap = va - tr

ib = int(np.argmin(va))
x_best, best_val = float(x[ib]), float(va[ib])
paciencia = int(rec.get('paciencia_validacoes') or 4)
un_pac = 'validações' if FRACAO else 'épocas'
_x = lambda z: (f'{z:.2f}'.replace('.', ',') if FRACAO else f'{int(z)}')
rotulo_best = f'época {_x(x_best)}'

# Diagnóstico automático a partir da curva (mesma regra nos dois formatos, por ponto de validação)
_pct = lambda z: f'{100 * z:.1f}'.replace('.', ',') + ' %'
va_d, tr_d = va[ib:], tr[ib:]
if len(va_d) <= 1:
  diag = f'O treino parou na melhor validação ({rotulo_best}): não há pontos depois dela para avaliar overfitting.'
elif va_d[-1] > best_val * 1.01 and np.isfinite(tr_d).all() and tr_d[-1] < tr_d[0]:
  diag = (f'Sinal de overfitting depois da {rotulo_best}: a perda de treino continuou caindo e a de validação '
          f'subiu {_pct(va_d[-1] / best_val - 1)}. O checkpoint guardado é o da {rotulo_best}, anterior a isso.')
else:
  diag = (f'Sem overfitting relevante: depois da {rotulo_best} a validação estacionou '
          f'(variação de {_pct(va_d.max() / best_val - 1)}) e a parada antecipada encerrou o treino.')
if x_best <= 2 and len(va_d) > 3:
  diag += ' A melhor validação veio muito cedo: o ajuste pode estar forte demais (taxa de aprendizado).'

# ------------------------------------------------------------------- figura
fig = plt.figure(figsize=(16, 7.2))
gs = fig.add_gridspec(1, 2, wspace=0.18)


def _fmt_t(s):
  try:
    s = float(s)
    return f'{s / 3600:.2f} h' if s >= 3600 else f'{s / 60:.1f} min'
  except (TypeError, ValueError):
    return 'não registrado'


tok = rec.get('tokenizacao_treino', 'não registrada (treino anterior à correção de 16/11/2026)')
origem = 'Célula 4.3 (promovido)' if rec.get('promovido_pela_celula_4_3') else 'Célula 4'
val_txt = f'validação a cada {VC:g} de época' if FRACAO else 'validação por época'
linhas = [
    f"Ambiente: {globals().get('ENV_LABEL', '?')}  |  GPU: {rec.get('gpu', '?')}  |  Precisão: {rec.get('precisao_numerica', '?')}"
    f"  |  Data do treino (UTC): {rec.get('data_treino_utc', '?')}  |  Origem: {origem}",
    f"Modelo: GPTCast-8x8 (Zenodo) — VQGAN congelada (90,71 M) + Transformer GPT-2 ajustado (304,93 M)  |  "
    f"Tokenização do treino: {tok}",
    f"Dados: IMERG-FR v7 a 2 km, RMF — treino 2019–2023 (9.940 janelas, corte de 8 quadros sorteado)  |  "
    f"validação 2024 (2.224 janelas, corte '{rec.get('corte_validacao', '?')}')",
    f"Otimização: AdamW (β 0,9/0,95, weight decay 0,01), lr {rec.get('learning_rate_efetiva', '?')} constante  |  "
    f"lote {rec.get('batch_size_treino', '?')} × acúmulo {rec.get('acumulo_gradiente', 1)}  |  semente {rec.get('semente_treino', '?')}"
    f"  |  {val_txt}  |  parada antecipada: paciência {paciencia} {un_pac} em val/loss",
    f"Resultado: {rec.get('epocas_executadas', '?')} épocas em {_fmt_t(rec.get('finetuning_tempo_total_s'))}  |  "
    f"melhor validação na {rotulo_best} (val/loss {best_val:.4f})  |  checkpoint: {rec.get('checkpoint', '?')}",
]
fig.suptitle('Curva de Treino do Fine-Tuning', fontsize=16, fontweight='bold', y=0.99)

# Gráfico 1: perdas
mk = 4 if FRACAO else 6
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(x, tr, 'o-', color='#1f77b4', lw=2, ms=mk,
         label='Treino (2019–2023)' + (' — média do trecho' if FRACAO else ''))
ax1.plot(x, va, 's-', color='#d62728', lw=2, ms=mk, label='Validação (2024)')
ax1.axvline(x_best, color='#2ca02c', ls='--', lw=1.5)
ax1.scatter([x_best], [best_val], s=160, facecolors='none', edgecolors='#2ca02c', lw=2.5, zorder=5,
            label=f'Melhor validação ({rotulo_best}): checkpoint guardado')
if x.max() > x_best:
  ax1.axvspan(x_best + passo_x / 2, x.max() + passo_x / 2, color='#ff7f0e', alpha=0.10,
              label=f'Validações sem melhora (paciência)' if FRACAO else 'Épocas sem melhora (paciência)')
tit = 'Perda ao longo do treino' if FRACAO else 'Perda por época'
ax1.set_title(tit + r'  $(\mathcal{L} = -\frac{1}{N}\,\Sigma_{i=1}^{N}\, \log p_\theta(z_i \mid z_{<i}))$', fontweight='bold')
ax1.set_xlabel('Época (fração)' if FRACAO else 'Época')
ax1.set_ylabel(r'$\mathcal{L}$ (entropia cruzada, nats)')
ax1.set_xticks(np.arange(1, int(np.ceil(x.max())) + 1) if FRACAO else x)
ax1.grid(True, ls=':', alpha=0.6)
ax1.legend(fontsize=8.5)

# Gráfico 2: diferença validação − treino
ax2 = fig.add_subplot(gs[0, 1])
cores = ['#d62728' if g > 0 else '#1f77b4' for g in gap]
ax2.bar(x, gap, width=0.8 * passo_x, color=cores, alpha=0.8)
ax2.axhline(0, color='k', lw=0.8)
ax2.axvline(x_best, color='#2ca02c', ls='--', lw=1.5, label=f'Melhor validação ({rotulo_best})')
ax2.set_title(r'Diferença validação − treino  $(\Delta = \mathcal{L}_{\mathrm{val}} - \mathcal{L}_{\mathrm{treino}})$', fontweight='bold')
ax2.set_xlabel('Época (fração)' if FRACAO else 'Época')
ax2.set_ylabel(r'$\Delta$')
ax2.set_xticks(np.arange(1, int(np.ceil(x.max())) + 1) if FRACAO else x)
ax2.grid(True, axis='y', ls=':', alpha=0.6)
ax2.legend(fontsize=8.5)

png = os.path.join(RESULTS_ENV_DIR, f"curva_finetuning_{globals().get('ENV_TAG', 'Ambiente_Ativo')}.png")
fig.savefig(png, dpi=200, bbox_inches='tight')
plt.show()

tabela = pd.DataFrame({'Época (fração)' if FRACAO else 'Época': [_x(z) for z in x],
                       'treino' + (' (trecho)' if FRACAO else ''): tr.round(4), 'validação': va.round(4),
                       'val − treino': gap.round(4)}).set_index('Época (fração)' if FRACAO else 'Época')
print('=== Contexto do treino (F-PluvIA Metropolitan — Célula 4.2) ===')
for ln in linhas:
  print('  ' + ln)
print(f'\n[Célula 4.2] Figura -> {png}\n')
print(tabela.to_string())
print('\n=== Diagnóstico automático (regras desta célula, não é um teste estatístico) ===')
print(textwrap.fill(diag, 110))
print(textwrap.fill(f'Regras: "sinal de overfitting" se a val/loss do último ponto de validação ficou mais de 1 % acima '
                    'da melhor e a perda de treino continuou caindo; "sem overfitting relevante" nos demais casos; aviso '
                    'extra se a melhor validação veio até o fim da 2ª época.'
                    + (' Com validação a cada fração de época, o treino em cada ponto é a média dos passos desde a '
                       'validação anterior.' if FRACAO else ''), 110))
