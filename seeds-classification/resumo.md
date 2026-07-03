# Resumo do Trabalho — Classificação de Qualidade de Sementes

**Disciplina:** Inteligência Computacional — CEFET-MG  
**Professor:** Alisson Marques da Silva  
**Tema:** Classificação supervisionada multiclasse de variedades de sementes de trigo

---

## Introdução

Este trabalho conduz um estudo experimental comparativo entre três técnicas de Inteligência Computacional aplicadas à **classificação de variedades de sementes de trigo** a partir de medidas morfológicas dos grãos. O problema é relevante na agricultura e na indústria alimentícia, pois a identificação correta da variedade impacta o controle de qualidade, a classificação comercial e o rastreamento de lotes.

A base utilizada foi o **Seeds Dataset** do repositório UCI Machine Learning Repository (id 236), composto por **210 amostras balanceadas** (70 por classe) e **7 atributos numéricos contínuos**: área, perímetro, compacidade, comprimento do núcleo, largura do núcleo, coeficiente de assimetria e comprimento do sulco da semente. A variável alvo é categórica, com três classes: **Kama**, **Rosa** e **Canadian**.

Foram comparados três classificadores supervisionados:

- **k-Nearest Neighbors (k-NN)** — classificador baseado em instâncias, sensível à escala dos atributos;
- **Árvore de Decisão** — modelo simbólico interpretável, invariante à escala;
- **Rede Neural Perceptron Multicamadas (MLP)** — modelo conexionista capaz de aprender fronteiras de decisão não lineares.

O foco do trabalho não foi apenas obter o melhor resultado possível, mas conduzir um **estudo experimental metodologicamente rigoroso**, com divisão adequada dos dados, ajuste sistemático de hiperparâmetros, comparação justa entre os modelos e análise estatística das diferenças de desempenho.

---

## Metodologia

### Dados e análise exploratória

O dataset foi carregado via pacote `ucimlrepo`, com fallback para arquivo local (`data/seeds_dataset.txt`). A análise exploratória confirmou:

- 210 amostras, 7 atributos numéricos contínuos, sem valores faltantes;
- balanceamento perfeito das classes (70/70/70);
- escalas muito diferentes entre atributos (ex.: área ~15 vs. compacidade ~0,88);
- alta correlação entre área, perímetro e comprimentos;
- separabilidade visual razoável na projeção PCA 2D, com leve sobreposição entre *Kama* e *Canadian*.

Foram gerados histogramas, boxplots, matriz de correlação, pairplot e projeção PCA 2D. A verificação de outliers pelo critério IQR identificou poucos pontos extremos (principalmente em compacidade e assimetria), mas optou-se por **não removê-los**, pois representam variabilidade natural e o conjunto é pequeno.

### Pré-processamento

- **Codificação do alvo:** `LabelEncoder` converteu as classes (1, 2, 3) para (0, 1, 2).
- **Holdout externo:** 20% dos dados (42 amostras) foram reservados como **conjunto de teste final**, de forma estratificada, **antes** de qualquer ajuste de hiperparâmetros.
- **Padronização:** `StandardScaler` foi aplicado via `Pipeline` apenas para k-NN e MLP, ajustado exclusivamente nos dados de treino de cada fold (evitando vazamento de dados). A Árvore de Decisão não recebeu padronização, por ser invariante à escala.
- **Reprodutibilidade:** `random_state=42` em todas as etapas estocásticas.

### Procedimento experimental

1. **Ajuste de hiperparâmetros:** `GridSearchCV` com validação cruzada estratificada **k=10** sobre os 168 exemplos de treino, testando sistematicamente todas as combinações de hiperparâmetros de cada modelo. A métrica de otimização foi **F1-macro** (adequada para 3 classes balanceadas).

2. **Validação cruzada repetida:** `RepeatedStratifiedKFold` com **k=10 e 10 repetições** (100 estimativas por modelo) para obter média ± desvio-padrão de acurácia e F1-macro. O mesmo objeto de CV (mesma semente) foi usado para os três modelos, garantindo comparação justa.

3. **Avaliação final:** cada modelo, com seus melhores hiperparâmetros, foi treinado em todo o conjunto de treino e avaliado **uma única vez** no holdout de teste.

4. **Testes estatísticos:** teste de **Friedman** (diferença global entre os modelos) seguido de post-hoc de **Nemenyi** (comparações par-a-par por ranks médios) e **Wilcoxon** pareado com correção de **Bonferroni** (verificação complementar).

### Hiperparâmetros testados

| Modelo | Parâmetros avaliados |
|---|---|
| k-NN | `n_neighbors` [3,5,7,9,11,15], `weights` [uniform, distance], `metric` [euclidean, manhattan, minkowski] |
| Árvore | `criterion` [gini, entropy], `max_depth` [3,5,7,10, None], `min_samples_split` [2,5,10], `min_samples_leaf` [1,2,4] |
| MLP | `hidden_layer_sizes` [(10,), (20,), (10,10), (20,10), (32,16)], `activation` [relu, tanh, logistic], `learning_rate_init` [0.001, 0.01, 0.1], `solver` [adam, lbfgs] |

### Métricas de avaliação

Acurácia, precisão, revocação e F1-score em **macro-média**, AUC (estratégia one-vs-rest), matriz de confusão, curvas ROC e tempo de treinamento.

---

## Resultados

### Melhores hiperparâmetros encontrados

| Modelo | Melhores hiperparâmetros | F1-macro (CV interna) |
|---|---|---|
| k-NN | metric=manhattan, n_neighbors=11, weights=distance | 0.9357 |
| Árvore de Decisão | criterion=entropy, max_depth=5, min_samples_leaf=1, min_samples_split=10 | 0.9343 |
| MLP | activation=logistic, hidden_layer_sizes=(10,), learning_rate_init=0.001, solver=lbfgs | 0.9821 |

### Desempenho comparativo

| Modelo | Acurácia (CV 10×10) | F1-macro (CV 10×10) | F1-macro (teste) | AUC (teste) | Tempo treino (s) |
|---|---|---|---|---|---|
| k-NN | 0.929 ± 0.056 | 0.928 ± 0.058 | 0.873 | 0.985 | 0.002 |
| Árvore de Decisão | 0.926 ± 0.069 | 0.925 ± 0.070 | **0.952** | 0.963 | 0.002 |
| **MLP** | **0.965 ± 0.047** | **0.964 ± 0.049** | 0.901 | 0.937 | 0.011 |

### Testes estatísticos

- **Friedman:** estatística = 39,28, **p ≈ 3×10⁻⁹** → diferença **estatisticamente significativa** entre os três modelos.
- **Nemenyi (ranks médios, menor = melhor):** MLP = 1,59, k-NN = 2,19, Árvore = 2,21. Diferença crítica (CD, α=0,05) = 0,331.
- **Wilcoxon + Bonferroni:** diferença significativa entre MLP e k-NN (p ≈ 5×10⁻⁷) e entre MLP e Árvore (p ≈ 1×10⁻⁵); k-NN vs. Árvore **não** significativa (p = 0,58).

### Principais observações

- Na **validação cruzada repetida** (estimativa mais robusta), o **MLP** obteve o melhor desempenho médio (F1-macro = 0,964 ± 0,049), seguido pelo k-NN e pela Árvore, com desempenhos próximos entre si.
- No **conjunto de teste** (apenas 42 amostras), a **Árvore de Decisão** liderou pontualmente (F1 = 0,952), enquanto o MLP caiu para 0,901 — reflexo da alta variância de um holdout pequeno.
- O k-NN apresentou a maior AUC no teste (0,985), indicando boa capacidade de ranqueamento probabilístico, apesar de um F1 ligeiramente inferior.
- Os erros de classificação concentraram-se principalmente na confusão entre as variedades **Kama** e **Canadian**, coerente com a sobreposição observada na PCA.
- O efeito dos hiperparâmetros foi analisado: no k-NN, valores intermediários de *k* com pesos por distância performaram melhor; na Árvore, `max_depth=5` evitou sobreajuste; no MLP, arquitetura simples (10 neurônios) com `lbfgs` e ativação logística foi suficiente para este dataset pequeno.

---

## Conclusão

O estudo comparou k-NN, Árvore de Decisão e MLP na classificação de três variedades de sementes de trigo, utilizando metodologia experimental rigorosa com holdout externo, validação cruzada repetida, ajuste sistemático de hiperparâmetros e testes estatísticos formais.

O **MLP** foi o modelo com melhor desempenho médio na validação cruzada repetida (F1-macro = 0,964 ± 0,049), e as diferenças entre os modelos foram **estatisticamente significativas** (Friedman, p < 0,001). O teste de Nemenyi confirmou a superioridade do MLP sobre k-NN e Árvore, enquanto estes dois últimos não diferiram significativamente entre si.

A **padronização** dos atributos mostrou-se indispensável para k-NN e MLP, dada a grande diferença de escalas entre os atributos morfológicos. A **Árvore de Decisão**, embora ligeiramente inferior na CV, oferece a vantagem da **interpretabilidade** e obteve o melhor resultado pontual no holdout de teste.

**Limitações:** o dataset é pequeno (210 amostras), o que torna as estimativas do conjunto de teste (42 amostras) instáveis; há alta correlação entre atributos; e as classes Kama e Canadian apresentam sobreposição parcial no espaço de atributos.

**Trabalhos futuros** sugeridos: métodos ensemble (Random Forest, Gradient Boosting), seleção de atributos para reduzir redundância, validação em bases maiores e comparação com técnicas de redução de dimensionalidade (LDA, t-SNE) para melhorar a separabilidade entre Kama e Canadian.

---

*Todos os gráficos, tabelas detalhadas e o código-fonte estão disponíveis em `results/` e `notebooks/analise_completa.ipynb`. Consulte o `README.md` para instruções de execução.*
