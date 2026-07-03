"""Análise Exploratória de Dados (EDA) do Seeds Dataset.

Fornece funções que geram:
- Resumo geral do dataset (nº de amostras, atributos, tipos, faltantes).
- Balanceamento das classes.
- Estatísticas descritivas globais e por classe.
- Gráficos: histogramas, boxplots por atributo/classe, matriz de correlação
  (heatmap), pairplot e projeção PCA 2D.

Todos os gráficos são salvos em `results/figures/` e todas as tabelas em
`results/tables/`.
"""

from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_loader import FEATURE_NAMES, TARGET_NAME
from utils import save_table, savefig

sns.set_theme(style="whitegrid", context="notebook")

# Paleta consistente por variedade, reutilizada em todos os gráficos.
CLASS_PALETTE = {"Kama": "#1f77b4", "Rosa": "#ff7f0e", "Canadian": "#2ca02c"}


def dataset_overview(df: pd.DataFrame) -> Dict[str, object]:
    """Monta um dicionário com o panorama geral do dataset e o imprime."""
    overview = {
        "n_amostras": int(df.shape[0]),
        "n_atributos": len(FEATURE_NAMES),
        "atributos": FEATURE_NAMES,
        "tipos_atributos": "todos numéricos contínuos (float64)",
        "variavel_alvo": TARGET_NAME,
        "n_classes": int(df[TARGET_NAME].nunique()),
        "valores_faltantes": int(df[FEATURE_NAMES].isna().sum().sum()),
        "duplicatas": int(df.duplicated(subset=FEATURE_NAMES).sum()),
    }

    print("=" * 60)
    print("PANORAMA GERAL DO DATASET")
    print("=" * 60)
    for k, v in overview.items():
        print(f"  {k:>20}: {v}")
    return overview


def class_balance(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Calcula e (opcionalmente) salva a distribuição das classes."""
    counts = df["variety_name"].value_counts().rename("n_amostras")
    prop = (df["variety_name"].value_counts(normalize=True) * 100).round(2)
    prop = prop.rename("percentual_%")
    balance = pd.concat([counts, prop], axis=1)
    balance.index.name = "variedade"

    if save:
        save_table(balance, "01_balanceamento_classes", index=True)
    return balance


def descriptive_stats(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Estatísticas descritivas globais (média, dp, min, quartis, max)."""
    stats = df[FEATURE_NAMES].describe().T
    stats = stats[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    if save:
        save_table(stats, "02_estatisticas_descritivas_global", index=True)
    return stats


def descriptive_stats_by_class(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Média e desvio-padrão de cada atributo por variedade.

    A visualização por classe é essencial para discutir a separabilidade:
    variedades cujas médias diferem bastante em vários atributos tendem a ser
    mais fáceis de classificar.
    """
    grouped = df.groupby("variety_name")[FEATURE_NAMES].agg(["mean", "std"])
    if save:
        # Achata o MultiIndex de colunas para salvar em CSV/MD de forma legível.
        flat = grouped.copy()
        flat.columns = [f"{feat}_{stat}" for feat, stat in flat.columns]
        save_table(flat, "03_estatisticas_por_classe", index=True)
    return grouped


def plot_histograms(df: pd.DataFrame, save: bool = True):
    """Histogramas com densidade (KDE) por classe para cada atributo."""
    n = len(FEATURE_NAMES)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.ravel()

    for i, feat in enumerate(FEATURE_NAMES):
        ax = axes[i]
        for cls, color in CLASS_PALETTE.items():
            subset = df[df["variety_name"] == cls][feat]
            sns.histplot(subset, ax=ax, color=color, label=cls, kde=True,
                         alpha=0.4, stat="density", bins=15)
        ax.set_title(feat)
        ax.set_xlabel("")
        ax.legend(fontsize=8)

    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Distribuição dos atributos por variedade", fontsize=14,
                 y=1.01)
    fig.tight_layout()
    if save:
        savefig(fig, "01_histogramas_por_classe")
    return fig


def plot_boxplots(df: pd.DataFrame, save: bool = True):
    """Boxplots de cada atributo agrupados por variedade."""
    n = len(FEATURE_NAMES)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.ravel()

    for i, feat in enumerate(FEATURE_NAMES):
        ax = axes[i]
        sns.boxplot(data=df, x="variety_name", y=feat, ax=ax,
                    palette=CLASS_PALETTE, order=list(CLASS_PALETTE))
        ax.set_title(feat)
        ax.set_xlabel("")

    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Boxplots dos atributos por variedade", fontsize=14, y=1.01)
    fig.tight_layout()
    if save:
        savefig(fig, "02_boxplots_por_classe")
    return fig


def plot_correlation(df: pd.DataFrame, save: bool = True):
    """Matriz de correlação (Pearson) entre os atributos como heatmap.

    Os coeficientes são exibidos e salvos em **porcentagem** (valor × 100),
    para facilitar a leitura no artigo (ex.: 99,4% em vez de 0,994).

    Correlações muito altas (ex.: área, perímetro e comprimento) indicam
    redundância de informação — relevante para discutir dimensionalidade e o
    comportamento dos modelos.
    """
    corr = df[FEATURE_NAMES].corr() * 100
    annot = corr.map(lambda v: f"{v:.1f}%")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=annot, fmt="", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8, "label": "%"},
                ax=ax)
    ax.set_title("Matriz de correlação dos atributos (%)")
    fig.tight_layout()
    if save:
        savefig(fig, "03_matriz_correlacao")
        save_table(corr, "04_matriz_correlacao", index=True, float_format="%.2f")
    return fig, corr


def plot_pairplot(df: pd.DataFrame, save: bool = True):
    """Pairplot (dispersão par-a-par) colorido por variedade."""
    g = sns.pairplot(
        df[FEATURE_NAMES + ["variety_name"]],
        hue="variety_name",
        palette=CLASS_PALETTE,
        corner=True,
        diag_kind="kde",
        plot_kws={"alpha": 0.6, "s": 25},
    )
    g.fig.suptitle("Pairplot dos atributos por variedade", y=1.02)
    if save:
        savefig(g.fig, "04_pairplot")
    return g


def plot_pca_2d(df: pd.DataFrame, save: bool = True):
    """Projeção PCA 2D (sobre dados padronizados) para visualizar separabilidade.

    A PCA é aplicada nos dados padronizados porque os atributos têm escalas
    muito diferentes; sem padronização, a área (valores ~15) dominaria os
    componentes principais.
    """
    X = df[FEATURE_NAMES].values
    X_std = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=0)
    comps = pca.fit_transform(X_std)
    var = pca.explained_variance_ratio_ * 100

    plot_df = pd.DataFrame(comps, columns=["PC1", "PC2"])
    plot_df["variedade"] = df["variety_name"].values

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="variedade",
                    palette=CLASS_PALETTE, s=60, alpha=0.8, ax=ax)
    ax.set_xlabel(f"PC1 ({var[0]:.1f}% da variância)")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% da variância)")
    ax.set_title(f"Projeção PCA 2D "
                 f"({var[0] + var[1]:.1f}% da variância total)")
    fig.tight_layout()
    if save:
        savefig(fig, "05_pca_2d")
    return fig, pca
