"""Funções auxiliares e utilitários do projeto.

Este módulo centraliza:
- Fixação de sementes aleatórias (reprodutibilidade total).
- Definição dos caminhos (paths) padrão do projeto.
- Funções auxiliares para salvar tabelas em CSV e Markdown.

A reprodutibilidade é um requisito central do trabalho: todas as etapas com
componente aleatória (splits, validação cruzada, inicialização de pesos do MLP)
usam a mesma semente base (RANDOM_STATE = 42).
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

# Semente global usada em todo o projeto. Mantê-la fixa garante que os
# experimentos possam ser reproduzidos exatamente.
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Caminhos do projeto
# ---------------------------------------------------------------------------
# Raiz do projeto = pasta que contém `src/`, `data/`, `results/`, etc.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = PROJECT_ROOT / "models"

# Caminho do dataset local (fallback caso o download via ucimlrepo falhe).
LOCAL_DATASET_PATH = DATA_DIR / "seeds_dataset.txt"


def ensure_dirs() -> None:
    """Garante que todas as pastas de saída existam."""
    for d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    """Fixa as sementes aleatórias de `random`, `numpy` e a variável de ambiente
    `PYTHONHASHSEED`, garantindo reprodutibilidade das etapas estocásticas.

    Observação: o scikit-learn não usa um estado global; por isso a semente
    também é passada explicitamente via `random_state` em cada estimador,
    split e validação cruzada ao longo do projeto.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def save_table(df: pd.DataFrame, name: str, index: bool = True,
               float_format: str = "%.4f") -> None:
    """Salva um DataFrame como CSV e como Markdown em `results/tables/`.

    Parameters
    ----------
    df : pd.DataFrame
        Tabela a ser salva.
    name : str
        Nome base do arquivo (sem extensão).
    index : bool
        Se o índice deve ser incluído nos arquivos.
    float_format : str
        Formato dos números de ponto flutuante.
    """
    ensure_dirs()
    csv_path = TABLES_DIR / f"{name}.csv"
    md_path = TABLES_DIR / f"{name}.md"

    df.to_csv(csv_path, index=index, float_format=float_format)

    # Markdown: útil para colar diretamente no artigo/relatório.
    try:
        md = df.to_markdown(index=index, floatfmt=".4f")
    except Exception:
        # Fallback caso `tabulate` não esteja instalado.
        md = df.to_string(index=index)
    md_path.write_text(md + "\n", encoding="utf-8")


def savefig(fig, name: str, dpi: int = 300) -> Path:
    """Salva uma figura matplotlib em alta resolução em `results/figures/`."""
    ensure_dirs()
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
