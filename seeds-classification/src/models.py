"""Definição dos modelos e das grades de hiperparâmetros.

Cada modelo é encapsulado em um `Pipeline` do scikit-learn. Para k-NN e MLP o
pipeline inclui um `StandardScaler`, de forma que a padronização seja ajustada
apenas com os dados de treino de cada fold da validação cruzada (evitando
vazamento). A Árvore de Decisão não recebe padronização, pois é invariante à
escala dos atributos.

As grades de hiperparâmetros seguem o enunciado do trabalho e permitem testar
sistematicamente várias configurações via `GridSearchCV`.

Breve descrição das técnicas
----------------------------
- k-NN: classificador baseado em instâncias; classifica cada amostra pela
  classe majoritária entre seus k vizinhos mais próximos. Não tem fase de
  treino propriamente dita; é sensível à escala e à escolha de k.
- Árvore de Decisão: particiona recursivamente o espaço de atributos com
  cortes que maximizam a pureza das classes (Gini/entropia). É interpretável e
  invariante à escala, mas propensa a overfitting sem controle de profundidade.
- MLP: rede neural feedforward treinada por retropropagação; aprende fronteiras
  de decisão não-lineares. Sensível à escala e à inicialização (aleatória) dos
  pesos, o que justifica repetições com múltiplas sementes.
"""

from __future__ import annotations

from typing import Dict

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from utils import RANDOM_STATE


def build_knn_pipeline() -> Pipeline:
    """Pipeline do k-NN: StandardScaler + KNeighborsClassifier."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier()),
    ])


def build_tree_pipeline() -> Pipeline:
    """Pipeline da Árvore de Decisão (sem padronização, por ser invariante à escala).

    Mantemos a estrutura de Pipeline por consistência de interface (mesmo tipo
    de objeto para os três modelos), mesmo sem etapa de escala.
    """
    return Pipeline([
        ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ])


def build_mlp_pipeline() -> Pipeline:
    """Pipeline do MLP: StandardScaler + MLPClassifier.

    `max_iter` alto e `early_stopping=True` ajudam na convergência e evitam
    overfitting; `random_state` fixo torna cada ajuste reprodutível.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            max_iter=2000,
            early_stopping=True,
            n_iter_no_change=25,
            random_state=RANDOM_STATE,
        )),
    ])


# ---------------------------------------------------------------------------
# Grades de hiperparâmetros (prefixo "clf__" por causa do Pipeline)
# ---------------------------------------------------------------------------
KNN_PARAM_GRID: Dict[str, list] = {
    "clf__n_neighbors": [3, 5, 7, 9, 11, 15],
    "clf__weights": ["uniform", "distance"],
    "clf__metric": ["euclidean", "manhattan", "minkowski"],
}

TREE_PARAM_GRID: Dict[str, list] = {
    "clf__criterion": ["gini", "entropy"],
    "clf__max_depth": [3, 5, 7, 10, None],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4],
}

# Para o MLP comparamos 'adam' e 'lbfgs'. Observação metodológica: o
# 'early_stopping' só é usado pelos solvers baseados em gradiente estocástico
# ('adam'/'sgd'); com 'lbfgs' ele é ignorado pelo scikit-learn. Ainda assim é
# válido comparar os dois solvers, pois o 'lbfgs' costuma performar bem em
# datasets pequenos.
MLP_PARAM_GRID: Dict[str, list] = {
    "clf__hidden_layer_sizes": [(10,), (20,), (10, 10), (20, 10), (32, 16)],
    "clf__activation": ["relu", "tanh", "logistic"],
    "clf__learning_rate_init": [0.001, 0.01, 0.1],
    "clf__solver": ["adam", "lbfgs"],
}


def get_models_and_grids() -> Dict[str, dict]:
    """Devolve, para cada modelo, seu pipeline e sua grade de hiperparâmetros."""
    return {
        "k-NN": {
            "pipeline": build_knn_pipeline(),
            "param_grid": KNN_PARAM_GRID,
        },
        "Árvore de Decisão": {
            "pipeline": build_tree_pipeline(),
            "param_grid": TREE_PARAM_GRID,
        },
        "MLP": {
            "pipeline": build_mlp_pipeline(),
            "param_grid": MLP_PARAM_GRID,
        },
    }
