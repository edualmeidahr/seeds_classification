"""Avaliação experimental: tuning, validação cruzada, métricas e testes estatísticos.

Este módulo concentra a metodologia experimental do trabalho:

1. `tune_model`         -> GridSearchCV com CV interna (k-fold estratificado)
                           sobre os dados de treino; registra TODAS as
                           configurações testadas.
2. `repeated_cv_scores` -> RepeatedStratifiedKFold para estimar acurácia e
                           F1-macro com múltiplas repetições (média ± desvio),
                           usando exatamente os mesmos folds para todos os
                           modelos (comparação justa).
3. `evaluate_on_test`   -> métricas no holdout final (acurácia, precisão,
                           revocação, F1-macro, AUC OvR, matriz de confusão,
                           classification_report).
4. Testes estatísticos  -> Friedman + post-hoc de Nemenyi (implementado com
                           `scipy.stats`, sem dependências externas).
5. Diversas funções de plotagem para as figuras exigidas.

Escolha das métricas
--------------------
Usamos F1-score, precisão e revocação em *macro-average* porque há 3 classes e
o dataset é balanceado (70/70/70): a macro-média trata todas as classes com o
mesmo peso, refletindo o desempenho equilibrado entre variedades (diferente da
micro/weighted, que seriam dominadas por eventuais classes majoritárias).
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.preprocessing import label_binarize

from utils import RANDOM_STATE, save_table, savefig

sns.set_theme(style="whitegrid", context="notebook")

MODEL_COLORS = {"k-NN": "#1f77b4", "Árvore de Decisão": "#ff7f0e", "MLP": "#2ca02c"}


# ---------------------------------------------------------------------------
# 1. Ajuste de hiperparâmetros (GridSearchCV)
# ---------------------------------------------------------------------------
def tune_model(
    name: str,
    pipeline,
    param_grid: dict,
    X_train,
    y_train,
    cv_splits: int = 10,
    scoring: str = "f1_macro",
    random_state: int = RANDOM_STATE,
) -> Tuple[GridSearchCV, pd.DataFrame]:
    """Executa GridSearchCV com CV estratificada interna sobre o treino.

    Returns
    -------
    (grid, results_df)
        `grid` é o objeto GridSearchCV já ajustado (com `best_estimator_`).
        `results_df` contém TODAS as configurações testadas e seus escores
        médios de validação (para discutir o impacto dos hiperparâmetros).
    """
    inner_cv = StratifiedKFold(n_splits=cv_splits, shuffle=True,
                               random_state=random_state)
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=inner_cv,
        n_jobs=-1,
        return_train_score=False,
    )
    grid.fit(X_train, y_train)

    cv_res = pd.DataFrame(grid.cv_results_)
    param_cols = [c for c in cv_res.columns if c.startswith("param_")]
    keep = param_cols + ["mean_test_score", "std_test_score", "rank_test_score"]
    results_df = (cv_res[keep]
                  .sort_values("rank_test_score")
                  .reset_index(drop=True))
    # Nomes de coluna mais limpos (sem o prefixo "param_clf__").
    results_df.columns = [c.replace("param_clf__", "").replace("param_", "")
                          for c in results_df.columns]

    print(f"[{name}] melhor {scoring} (CV interna) = "
          f"{grid.best_score_:.4f}")
    print(f"[{name}] melhores hiperparâmetros: {grid.best_params_}")
    return grid, results_df


# ---------------------------------------------------------------------------
# 2. Validação cruzada repetida (estimativa robusta + comparação justa)
# ---------------------------------------------------------------------------
def repeated_cv_scores(
    fitted_models: Dict[str, object],
    X,
    y,
    n_splits: int = 10,
    n_repeats: int = 10,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Avalia os modelos (já com melhores hiperparâmetros) via RepeatedStratifiedKFold.

    O MESMO objeto `RepeatedStratifiedKFold` (mesma semente) é usado para todos
    os modelos, garantindo que sejam avaliados exatamente sobre os mesmos
    folds — condição necessária para os testes estatísticos pareados.

    Returns
    -------
    (summary_df, per_fold_scores)
        `summary_df`: média ± desvio de acurácia e F1-macro por modelo.
        `per_fold_scores`: dicionário {modelo: array de F1-macro por fold},
        usado nos testes estatísticos e no boxplot.
    """
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=random_state)
    scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro"}

    rows = []
    per_fold_f1: Dict[str, np.ndarray] = {}
    per_fold_acc: Dict[str, np.ndarray] = {}

    for name, model in fitted_models.items():
        cvres = cross_validate(
            clone(model), X, y,
            cv=rskf, scoring=scoring, n_jobs=-1,
        )
        acc = cvres["test_accuracy"]
        f1 = cvres["test_f1_macro"]
        per_fold_acc[name] = acc
        per_fold_f1[name] = f1
        rows.append({
            "modelo": name,
            "acuracia_media": acc.mean(),
            "acuracia_std": acc.std(),
            "f1_macro_media": f1.mean(),
            "f1_macro_std": f1.std(),
        })

    summary_df = pd.DataFrame(rows).set_index("modelo")
    return summary_df, {"f1_macro": per_fold_f1, "accuracy": per_fold_acc}


# ---------------------------------------------------------------------------
# 3. Avaliação final no holdout de teste
# ---------------------------------------------------------------------------
def evaluate_on_test(
    name: str,
    model,
    X_test,
    y_test,
    class_names: List[str],
) -> Dict[str, object]:
    """Calcula todas as métricas finais para um modelo já ajustado no treino."""
    y_pred = model.predict(X_test)

    # Probabilidades (necessárias para ROC/AUC). Todos os 3 estimadores expõem
    # predict_proba (o KNN e a árvore sempre; o MLP também).
    y_proba = model.predict_proba(X_test)

    metrics = {
        "modelo": name,
        "acuracia": accuracy_score(y_test, y_pred),
        "precisao_macro": precision_score(y_test, y_pred, average="macro"),
        "revocacao_macro": recall_score(y_test, y_pred, average="macro"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "auc_ovr_macro": roc_auc_score(y_test, y_proba, multi_class="ovr",
                                       average="macro"),
    }
    report = classification_report(y_test, y_pred, target_names=class_names,
                                   digits=4)
    cm = confusion_matrix(y_test, y_pred)

    metrics["_report"] = report
    metrics["_confusion_matrix"] = cm
    metrics["_y_pred"] = y_pred
    metrics["_y_proba"] = y_proba
    return metrics


def measure_training_time(model, X_train, y_train, n_repeats: int = 5) -> float:
    """Mede o tempo médio de treino (fit) em segundos, com clones do modelo."""
    times = []
    for _ in range(n_repeats):
        m = clone(model)
        t0 = time.perf_counter()
        m.fit(X_train, y_train)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))


# ---------------------------------------------------------------------------
# 4. Testes estatísticos: Friedman + post-hoc de Nemenyi
# ---------------------------------------------------------------------------
def friedman_test(per_fold_scores: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Teste de Friedman: H0 = todos os modelos têm desempenho equivalente.

    Recebe os escores por fold (mesmos folds para todos os modelos).
    """
    names = list(per_fold_scores.keys())
    arrays = [per_fold_scores[n] for n in names]
    stat, p = stats.friedmanchisquare(*arrays)
    return {"statistic": float(stat), "p_value": float(p),
            "n_modelos": len(names), "n_amostras": len(arrays[0])}


def nemenyi_posthoc(per_fold_scores: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Post-hoc de Nemenyi baseado nos ranks médios.

    Implementação própria (usa a distribuição do studentized range via
    `scipy.stats.studentized_range`), evitando dependência do `scikit-posthocs`.

    Returns
    -------
    (pvalues_df, avg_ranks_df, critical_difference)
        matriz de p-valores par-a-par, ranks médios por modelo, e a diferença
        crítica (CD) para o diagrama de significância (alfa=0.05).
    """
    names = list(per_fold_scores.keys())
    # Matriz N x k (linhas = folds, colunas = modelos).
    data = np.column_stack([per_fold_scores[n] for n in names])
    n_datasets, k = data.shape

    # Ranks por linha: maior escore -> melhor -> rank 1 (por isso negamos).
    ranks = np.apply_along_axis(stats.rankdata, 1, -data)
    avg_ranks = ranks.mean(axis=0)

    # Estatística q = (R_i - R_j) / sqrt(k(k+1)/(6N)).
    se = np.sqrt(k * (k + 1) / (6.0 * n_datasets))

    pvals = np.ones((k, k))
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            q = abs(avg_ranks[i] - avg_ranks[j]) / se
            # p-valor pela distribuição do studentized range (k grupos, inf df).
            p = stats.studentized_range.sf(q * np.sqrt(2), k, np.inf)
            pvals[i, j] = min(1.0, p)

    pvalues_df = pd.DataFrame(pvals, index=names, columns=names)
    avg_ranks_df = (pd.Series(avg_ranks, index=names, name="rank_medio")
                    .sort_values().to_frame())

    # Diferença crítica (Demšar, 2006), alfa=0.05.
    q_alpha = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n_datasets))

    return pvalues_df, avg_ranks_df, float(cd)


def wilcoxon_bonferroni(per_fold_scores: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Wilcoxon pareado entre todos os pares, com correção de Bonferroni.

    Fornecido como teste alternativo/complementar ao Nemenyi.
    """
    names = list(per_fold_scores.keys())
    pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]
    m = len(pairs)  # nº de comparações (para Bonferroni)
    rows = []
    for i, j in pairs:
        a, b = per_fold_scores[names[i]], per_fold_scores[names[j]]
        try:
            stat, p = stats.wilcoxon(a, b)
        except ValueError:
            # Ocorre quando todas as diferenças são zero (modelos idênticos).
            stat, p = np.nan, 1.0
        rows.append({
            "comparacao": f"{names[i]} vs {names[j]}",
            "statistic": stat,
            "p_value": p,
            "p_value_bonferroni": min(1.0, p * m),
            "significativo_5%": (p * m) < 0.05,
        })
    return pd.DataFrame(rows).set_index("comparacao")


# ---------------------------------------------------------------------------
# 5. Funções de plotagem
# ---------------------------------------------------------------------------
def plot_cv_boxplot(per_fold_scores: Dict[str, np.ndarray], metric_name: str,
                    save_name: str, save: bool = True):
    """Boxplot comparativo dos escores de CV entre os modelos."""
    df = pd.DataFrame(per_fold_scores)
    fig, ax = plt.subplots(figsize=(8, 6))
    order = list(df.columns)
    sns.boxplot(data=df, order=order, palette=MODEL_COLORS, ax=ax)
    sns.stripplot(data=df, order=order, color="black", size=3, alpha=0.4, ax=ax)
    ax.set_ylabel(metric_name)
    ax.set_title(f"Distribuição de {metric_name} na validação cruzada repetida")
    fig.tight_layout()
    if save:
        savefig(fig, save_name)
    return fig


def plot_confusion_matrices(results: Dict[str, dict], class_names: List[str],
                            save: bool = True):
    """Matrizes de confusão dos modelos no conjunto de teste (heatmaps)."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, results.items()):
        cm = res["_confusion_matrix"]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=class_names)
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(f"{name}\n(acurácia = {res['acuracia']:.3f})")
    fig.suptitle("Matrizes de confusão — conjunto de teste", y=1.03,
                 fontsize=14)
    fig.tight_layout()
    if save:
        savefig(fig, "07_matrizes_confusao")
    return fig


def plot_roc_curves(results: Dict[str, dict], y_test, class_names: List[str],
                    save: bool = True):
    """Curvas ROC one-vs-rest (uma curva macro por modelo, e por classe)."""
    classes = np.arange(len(class_names))
    y_bin = label_binarize(y_test, classes=classes)

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5),
                             sharey=True)
    if len(results) == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        y_proba = res["_y_proba"]
        # Curva por classe.
        for c in classes:
            fpr, tpr, _ = roc_curve(y_bin[:, c], y_proba[:, c])
            ax.plot(fpr, tpr, lw=1.5,
                    label=f"{class_names[c]} (AUC={auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
        ax.set_title(f"{name}\nAUC macro OvR = {res['auc_ovr_macro']:.3f}")
        ax.set_xlabel("Taxa de falsos positivos")
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("Taxa de verdadeiros positivos")
    fig.suptitle("Curvas ROC (one-vs-rest) — conjunto de teste", y=1.03,
                 fontsize=14)
    fig.tight_layout()
    if save:
        savefig(fig, "08_curvas_roc")
    return fig


def plot_final_metrics_bars(summary_cv: pd.DataFrame, save: bool = True):
    """Gráfico de barras com erro (mean ± std) de acurácia e F1 (CV repetida)."""
    models = summary_cv.index.tolist()
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width / 2, summary_cv["acuracia_media"], width,
           yerr=summary_cv["acuracia_std"], capsize=5, label="Acurácia",
           color="#4c72b0")
    ax.bar(x + width / 2, summary_cv["f1_macro_media"], width,
           yerr=summary_cv["f1_macro_std"], capsize=5, label="F1-macro",
           color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("Escore (média ± desvio)")
    ax.set_title("Comparação de desempenho (validação cruzada repetida)")
    ax.legend()
    fig.tight_layout()
    if save:
        savefig(fig, "06_barras_metricas_cv")
    return fig


def plot_knn_hyperparam(grid_results: pd.DataFrame, save: bool = True):
    """Acurácia/F1 vs n_neighbors para o k-NN (média sobre demais params)."""
    df = grid_results.copy()
    df["n_neighbors"] = df["n_neighbors"].astype(int)
    fig, ax = plt.subplots(figsize=(8, 5))
    weights = df["weights"].unique() if "weights" in df.columns else ["(todos)"]
    for weight in weights:
        sub_df = df if weight == "(todos)" else df[df["weights"] == weight]
        sub = sub_df.groupby("n_neighbors")["mean_test_score"].mean()
        ax.plot(sub.index, sub.values, marker="o", label=f"weights={weight}")
    ax.set_xlabel("n_neighbors (k)")
    ax.set_ylabel("F1-macro médio (CV interna)")
    ax.set_title("k-NN: efeito de k e do esquema de pesos")
    ax.legend()
    fig.tight_layout()
    if save:
        savefig(fig, "09_hiperparam_knn")
    return fig


def plot_tree_hyperparam(grid_results: pd.DataFrame, save: bool = True):
    """F1 vs max_depth para a Árvore de Decisão (por critério)."""
    df = grid_results.copy()
    # None -> um valor grande apenas para posicionar no eixo.
    df["max_depth_plot"] = df["max_depth"].apply(
        lambda v: 15 if (v is None or (isinstance(v, float) and np.isnan(v))
                         or str(v) == "None") else int(v))
    fig, ax = plt.subplots(figsize=(8, 5))
    crits = df["criterion"].unique() if "criterion" in df.columns else ["(todos)"]
    for crit in crits:
        sub_df = df if crit == "(todos)" else df[df["criterion"] == crit]
        sub = sub_df.groupby("max_depth_plot")["mean_test_score"].mean()
        ax.plot(sub.index, sub.values, marker="o", label=f"criterion={crit}")
    ax.set_xlabel("max_depth (15 = None/ilimitada)")
    ax.set_ylabel("F1-macro médio (CV interna)")
    ax.set_title("Árvore de Decisão: efeito da profundidade máxima")
    ax.legend()
    fig.tight_layout()
    if save:
        savefig(fig, "10_hiperparam_arvore")
    return fig


def plot_mlp_hyperparam(grid_results: pd.DataFrame, save: bool = True):
    """F1 vs arquitetura (hidden_layer_sizes) para o MLP (por ativação)."""
    df = grid_results.copy()
    df["arquitetura"] = df["hidden_layer_sizes"].astype(str)
    # Ordena arquiteturas pelo número total de neurônios para leitura no eixo.
    def n_neurons(s):
        nums = [int(x) for x in s.strip("()").replace(" ", "").split(",") if x]
        return sum(nums)
    order = sorted(df["arquitetura"].unique(), key=n_neurons)

    fig, ax = plt.subplots(figsize=(9, 5))
    acts = df["activation"].unique() if "activation" in df.columns else ["(todos)"]
    for act in acts:
        sub_df = df if act == "(todos)" else df[df["activation"] == act]
        sub = sub_df.groupby("arquitetura")["mean_test_score"].mean()
        sub = sub.reindex(order)
        ax.plot(range(len(order)), sub.values, marker="o", label=f"activation={act}")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_xlabel("Arquitetura (hidden_layer_sizes)")
    ax.set_ylabel("F1-macro médio (CV interna)")
    ax.set_title("MLP: efeito da arquitetura e da função de ativação")
    ax.legend()
    fig.tight_layout()
    if save:
        savefig(fig, "11_hiperparam_mlp")
    return fig


def plot_mlp_loss_curve(mlp_pipeline, X_train, y_train, save: bool = True):
    """Curva de perda (loss) do MLP ao longo das épocas.

    Reajusta o melhor MLP forçando o solver 'adam' (que expõe `loss_curve_`),
    apenas para fins de visualização da convergência. Se o melhor solver já for
    'adam', a curva reflete o modelo final; se for 'lbfgs', esta é uma
    ilustração da dinâmica de treino com adam.
    """
    mlp = clone(mlp_pipeline)
    clf = mlp.named_steps["clf"]
    if clf.get_params().get("solver") == "lbfgs":
        mlp.set_params(clf__solver="adam")
    mlp.set_params(clf__early_stopping=False)  # curva completa
    mlp.fit(X_train, y_train)
    loss = mlp.named_steps["clf"].loss_curve_

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(loss) + 1), loss, color=MODEL_COLORS["MLP"])
    ax.set_xlabel("Época")
    ax.set_ylabel("Perda (log-loss)")
    ax.set_title("Curva de aprendizado do MLP (treino)")
    fig.tight_layout()
    if save:
        savefig(fig, "12_curva_loss_mlp")
    return fig


def plot_decision_tree(tree_pipeline, feature_names, class_names,
                       save: bool = True):
    """Visualização da Árvore de Decisão treinada (opcional)."""
    from sklearn.tree import plot_tree
    clf = tree_pipeline.named_steps["clf"]
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(clf, feature_names=feature_names, class_names=class_names,
              filled=True, rounded=True, fontsize=9, ax=ax)
    ax.set_title("Árvore de Decisão treinada (melhores hiperparâmetros)")
    fig.tight_layout()
    if save:
        savefig(fig, "13_arvore_decisao")
    return fig


def plot_nemenyi_cd(avg_ranks_df: pd.DataFrame, cd: float, save: bool = True):
    """Diagrama de diferença crítica (CD) do teste de Nemenyi."""
    ranks = avg_ranks_df["rank_medio"]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    y = 1
    ax.scatter(ranks.values, [y] * len(ranks), s=60, color="black", zorder=3)
    for name, r in ranks.items():
        ax.annotate(name, (r, y), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=10)
    # Barra da diferença crítica.
    r_min = ranks.min()
    ax.plot([r_min, r_min + cd], [y - 0.25, y - 0.25], color="red", lw=3)
    ax.annotate(f"CD = {cd:.2f}", ((r_min + r_min + cd) / 2, y - 0.25),
                xytext=(0, -18), textcoords="offset points", ha="center",
                color="red", fontsize=10)
    ax.set_xlabel("Rank médio (menor = melhor)")
    ax.set_yticks([])
    ax.set_ylim(0.4, 1.4)
    ax.set_title("Teste de Nemenyi — diagrama de diferença crítica (α=0.05)")
    fig.tight_layout()
    if save:
        savefig(fig, "14_nemenyi_cd")
    return fig
