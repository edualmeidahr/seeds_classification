"""Pré-processamento dos dados.

Responsabilidades:
- Verificação de dados faltantes e de outliers (documentação da verificação).
- Codificação do alvo com `LabelEncoder` (mapeamento documentado).
- Separação de um conjunto de TESTE externo (holdout) estratificado, feito
  ANTES de qualquer ajuste de hiperparâmetros, para evitar vazamento de dados.

Decisão de projeto sobre padronização
--------------------------------------
A padronização (StandardScaler) é necessária para k-NN e MLP, que são sensíveis
à escala dos atributos (o k-NN usa distâncias; o MLP usa gradiente/pesos). A
Árvore de Decisão é invariante a transformações monotônicas de cada atributo e,
portanto, não requer padronização.

Para garantir uma comparação justa e evitar vazamento de dados, a padronização
NÃO é aplicada uma única vez sobre todo o conjunto. Em vez disso, ela é embutida
em um `Pipeline` (scikit-learn) para k-NN e MLP, de modo que o `StandardScaler`
seja ajustado apenas nos dados de treino de cada fold da validação cruzada. Este
módulo apenas fornece os utilitários; a construção dos pipelines fica em
`models.py`.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from data_loader import FEATURE_NAMES
from utils import RANDOM_STATE


def check_missing(df: pd.DataFrame) -> pd.Series:
    """Retorna a contagem de valores faltantes por coluna."""
    return df[FEATURE_NAMES].isna().sum()


def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Detecta outliers por atributo pelo critério do IQR (Tukey, 1.5*IQR).

    Retorna uma tabela com o número de outliers por atributo. O objetivo aqui
    é *documentar* a verificação; optamos por NÃO remover os outliers, pois em
    um dataset pequeno (210 amostras) e sem erros evidentes de medição, remover
    pontos extremos poderia descartar variabilidade natural das sementes e
    reduzir a robustez dos modelos.
    """
    rows = []
    for feat in FEATURE_NAMES:
        q1 = df[feat].quantile(0.25)
        q3 = df[feat].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        mask = (df[feat] < low) | (df[feat] > high)
        rows.append({
            "atributo": feat,
            "n_outliers": int(mask.sum()),
            "limite_inferior": round(low, 4),
            "limite_superior": round(high, 4),
        })
    return pd.DataFrame(rows).set_index("atributo")


def encode_target(y: pd.Series) -> Tuple[np.ndarray, LabelEncoder, Dict[int, str]]:
    """Codifica o alvo com LabelEncoder e devolve o mapeamento documentado.

    O alvo original já é numérico (1, 2, 3). O LabelEncoder o converte para
    (0, 1, 2), formato esperado por várias funções do scikit-learn (ex.:
    curvas ROC one-vs-rest).
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    mapping = {int(enc): int(orig) for enc, orig in enumerate(le.classes_)}
    return y_enc, le, mapping


def make_holdout_split(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Separa um conjunto de teste externo estratificado (holdout).

    Este conjunto de teste é reservado ANTES de qualquer ajuste de
    hiperparâmetros e usado apenas na avaliação final, garantindo uma
    estimativa não-enviesada da capacidade de generalização.

    A estratificação (`stratify=y`) preserva a proporção 1/3 de cada classe
    tanto no treino quanto no teste.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test
