# Classificação de Qualidade de Sementes (Seeds Dataset)

Estudo experimental comparativo entre três técnicas de Inteligência Computacional
aplicadas à **classificação supervisionada multiclasse** de variedades de sementes de
trigo: **k-Nearest Neighbors (k-NN)**, **Árvore de Decisão** e **Rede Neural Perceptron
Multicamadas (MLP)**.

> Trabalho final da disciplina **Inteligência Computacional** — CEFET-MG —
> Prof. Alisson Marques da Silva.

---

## 1. Problema e dados

- **Tarefa:** classificação multiclasse (3 classes).
- **Base de dados:** [Seeds Dataset — UCI Machine Learning Repository (id 236)](https://archive.ics.uci.edu/dataset/236/seeds).
- **Amostras:** 210 (balanceadas: 70 por classe).
- **Atributos:** 7 numéricos contínuos — área, perímetro, compacidade, comprimento do
  núcleo, largura do núcleo, coeficiente de assimetria e comprimento do sulco.
- **Classes (variedades):** Kama, Rosa, Canadian.
- **Dados faltantes:** nenhum.

---

## 2. Estrutura do projeto

```
seeds-classification/
├── README.md                    # este arquivo
├── requirements.txt             # dependências
├── data/
│   └── seeds_dataset.txt        # dataset local (fallback do download UCI)
├── src/
│   ├── utils.py                 # seeds, caminhos, salvamento de tabelas/figuras
│   ├── data_loader.py           # download (ucimlrepo) + fallback local
│   ├── eda.py                   # análise exploratória (estatísticas e gráficos)
│   ├── preprocessing.py         # outliers, LabelEncoder, holdout estratificado
│   ├── models.py                # pipelines dos 3 modelos + grades de hiperparâmetros
│   └── evaluation.py            # tuning, CV repetida, métricas, testes estatísticos, plots
├── notebooks/
│   └── analise_completa.ipynb   # ENTREGÁVEL PRINCIPAL (executável de ponta a ponta)
├── results/
│   ├── figures/                 # 16 gráficos em .png (alta resolução)
│   └── tables/                  # tabelas de resultados em .csv e .md
└── models/                      # modelos treinados (.joblib)
```

---

## 3. Instalação

Requer **Python 3.10+**.

```bash
cd seeds-classification
python -m venv .venv && source .venv/bin/activate   # opcional
pip install -r requirements.txt
```

> **Observação:** o pacote opcional `scikit-posthocs` **não é necessário**. O teste
> post-hoc de Nemenyi é implementado internamente com `scipy.stats` (distribuição do
> *studentized range*). Se `scikit-posthocs` estiver instalado, nada muda no resultado.

---

## 4. Como executar

### Opção A — Notebook (recomendado, é o entregável principal)

```bash
jupyter notebook notebooks/analise_completa.ipynb
```

Execute todas as células (menu *Run → Run All*). O notebook roda de ponta a ponta e
regenera automaticamente todas as figuras (`results/figures/`), tabelas
(`results/tables/`) e modelos (`models/`).

Para executar em modo não-interativo (linha de comando):

```bash
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 notebooks/analise_completa.ipynb
```

Tempo típico de execução completa: **~45 segundos** (grades de hiperparâmetros +
validação cruzada repetida 10×10).

### Opção B — Módulos `src/`

Os módulos em `src/` são reutilizáveis e podem ser importados em qualquer script:

```python
import sys; sys.path.insert(0, "src")
from utils import set_global_seed
import data_loader, eda, preprocessing, models, evaluation
set_global_seed(42)
df = data_loader.load_seeds()
# ... (ver o fluxo completo no notebook)
```

---

## 5. Metodologia experimental (resumo)

1. **Holdout externo:** 20% dos dados são reservados (estratificado) para **teste
   final**, **antes** de qualquer ajuste — nunca usados no tuning.
2. **Tuning:** `GridSearchCV` com **validação cruzada estratificada (k=10)** sobre o
   treino, testando **todas** as combinações de hiperparâmetros de cada modelo.
3. **Estimativa robusta:** `RepeatedStratifiedKFold` **(10×10 = 100 estimativas)** para
   média ± desvio de acurácia e F1-macro.
4. **Comparação justa:** os três modelos são avaliados exatamente sobre os **mesmos
   folds** (mesma semente).
5. **Teste estatístico:** **Friedman** + post-hoc de **Nemenyi** (com diferença crítica),
   e **Wilcoxon** pareado com correção de **Bonferroni** como verificação.
6. **Padronização:** `StandardScaler` embutido em `Pipeline` (ajustado só no treino de
   cada fold, evitando vazamento) para **k-NN** e **MLP**; a **Árvore** dispensa por ser
   invariante à escala.
7. **Reprodutibilidade:** `random_state=42` em todas as etapas estocásticas.

**Métricas:** acurácia, precisão/revocação/F1 em **macro-média** (3 classes
balanceadas), AUC (one-vs-rest), matriz de confusão e curvas ROC.

---

## 6. Principais resultados

| Modelo | Acurácia (CV 10×10) | F1-macro (CV 10×10) | F1-macro (teste) | AUC (teste) |
|---|---|---|---|---|
| k-NN | 0.929 ± 0.056 | 0.928 ± 0.058 | 0.873 | 0.985 |
| Árvore de Decisão | 0.926 ± 0.069 | 0.925 ± 0.070 | **0.952** | 0.963 |
| MLP | **0.965 ± 0.047** | **0.964 ± 0.049** | 0.901 | 0.937 |

- Na estimativa robusta (CV repetida), o **MLP** obteve o melhor desempenho médio.
- O teste de **Friedman** indicou diferença **estatisticamente significativa**
  (p ≈ 3×10⁻⁹); pelos ranks médios (Nemenyi) o MLP ficou em primeiro.
- No holdout de teste (42 amostras) a Árvore liderou pontualmente — reflexo da alta
  variância de um conjunto de teste pequeno; por isso a **CV repetida** é a estimativa
  mais confiável.

Consulte `results/tables/14_resumo_resultados.md` para o resumo textual completo e
`results/figures/` para todos os gráficos.

---

## 7. Reprodutibilidade

Todos os resultados são determinísticos (semente global `42`). Reexecutar o notebook
regenera figuras, tabelas e modelos idênticos.

---

## 8. Referência do dataset

Charytanowicz, M., Niewczas, J., Kulczycki, P., Kowalski, P., Łukasik, S., & Żak, S.
(2010). *Seeds* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5H30K
