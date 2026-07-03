"""Carregamento e organização do Seeds Dataset (UCI).

Estratégia de carga (robusta):
1. Tenta baixar o dataset diretamente do repositório UCI via `ucimlrepo`.
2. Em caso de falha (sem internet, API fora do ar, etc.), faz o fallback para
   o arquivo local `data/seeds_dataset.txt`.

O dataset original (`seeds_dataset.txt`) é separado por tabulações, mas contém
tabulações duplicadas em algumas linhas; por isso a leitura usa uma expressão
regular de espaços em branco (`\\s+`) como separador.

Referência:
    Charytanowicz, M., Niewczas, J., Kulczycki, P., Kowalski, P., Lukasik, S.,
    & Zak, S. (2010). Seeds [Dataset]. UCI Machine Learning Repository.
    https://doi.org/10.24432/C5H30K
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from utils import LOCAL_DATASET_PATH

# Nomes dos atributos (em inglês, como na documentação do UCI) e alvo.
FEATURE_NAMES: List[str] = [
    "area",              # área A
    "perimeter",         # perímetro P
    "compactness",       # compacidade C = 4*pi*A / P^2
    "kernel_length",     # comprimento do núcleo
    "kernel_width",      # largura do núcleo
    "asymmetry_coeff",   # coeficiente de assimetria
    "kernel_groove_length",  # comprimento do sulco do núcleo
]

TARGET_NAME = "variety"

# Mapeamento do código numérico original (1, 2, 3) para o nome da variedade.
CLASS_MAPPING = {1: "Kama", 2: "Rosa", 3: "Canadian"}


def _load_from_local() -> pd.DataFrame:
    """Lê o dataset do arquivo local, tratando o separador irregular."""
    df = pd.read_csv(
        LOCAL_DATASET_PATH,
        sep=r"\s+",
        header=None,
        names=FEATURE_NAMES + [TARGET_NAME],
        engine="python",
    )
    df[TARGET_NAME] = df[TARGET_NAME].astype(int)
    return df


def _load_from_ucimlrepo() -> pd.DataFrame:
    """Tenta baixar o dataset via pacote `ucimlrepo` (id=236)."""
    from ucimlrepo import fetch_ucirepo

    seeds = fetch_ucirepo(id=236)
    X = seeds.data.features.copy()
    y = seeds.data.targets.copy()

    # Padroniza os nomes das colunas para o padrão interno do projeto.
    X.columns = FEATURE_NAMES[: X.shape[1]]
    target_col = y.columns[0]
    df = X.copy()
    df[TARGET_NAME] = y[target_col].astype(int).values
    return df


def load_seeds(prefer_online: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Carrega o Seeds Dataset como um DataFrame.

    Parameters
    ----------
    prefer_online : bool
        Se True, tenta primeiro o download via `ucimlrepo`; caso contrário
        (ou em caso de falha), usa o arquivo local.
    verbose : bool
        Imprime a fonte efetivamente utilizada.

    Returns
    -------
    pd.DataFrame
        Colunas: 7 atributos numéricos + coluna `variety` (int 1..3) +
        coluna `variety_name` (str).
    """
    df = None
    source = None

    if prefer_online:
        try:
            df = _load_from_ucimlrepo()
            source = "UCI (ucimlrepo)"
        except Exception as exc:  # noqa: BLE001 - fallback intencional
            if verbose:
                print(f"[data_loader] Falha ao baixar via ucimlrepo ({exc!r}). "
                      f"Usando arquivo local.")

    if df is None:
        df = _load_from_local()
        source = f"arquivo local ({LOCAL_DATASET_PATH.name})"

    # Coluna textual com o nome da classe (útil para EDA e gráficos).
    df["variety_name"] = df[TARGET_NAME].map(CLASS_MAPPING)

    if verbose:
        print(f"[data_loader] Dataset carregado de: {source}")
        print(f"[data_loader] Formato: {df.shape[0]} amostras x "
              f"{len(FEATURE_NAMES)} atributos + alvo")

    return df


def split_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa o DataFrame em matriz de atributos X e vetor alvo y (numérico)."""
    X = df[FEATURE_NAMES].copy()
    y = df[TARGET_NAME].copy()
    return X, y
