"""
Visualisation et évaluation des résultats de détection d'anomalies.

Fonctions utilisées dans les notebooks et scripts pour interpréter les scores.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    precision_recall_curve,
)

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


# ─── Visualisation des scores d'anomalie ─────────────────────────────────────

def plot_anomaly_score_distribution(
    scores: pd.Series,
    threshold: float | None = None,
    title: str = "Distribution des scores d'anomalie",
    ax=None,
) -> None:
    """Histogramme des scores avec seuil optionnel."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    if threshold is not None:
        ax.axvline(threshold, color="red", linestyle="--", label=f"Seuil = {threshold:.3f}")
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Score")
    ax.set_ylabel("Nombre de samples")
    plt.tight_layout()


def plot_audiogram(
    dots_left: dict,
    dots_right: dict,
    title: str = "Audiogramme",
    ax=None,
) -> None:
    """
    Trace un audiogramme standard avec les seuils OG (×) et OD (○).

    L'axe Y est inversé (convention audiométrique : 0 dB en haut).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    if dots_left:
        freqs_l = sorted(dots_left.keys())
        dbs_l = [dots_left[f] for f in freqs_l]
        ax.plot(freqs_l, dbs_l, "x-", color="blue", label="OG (gauche)", markersize=10, linewidth=1.5)

    if dots_right:
        freqs_r = sorted(dots_right.keys())
        dbs_r = [dots_right[f] for f in freqs_r]
        ax.plot(freqs_r, dbs_r, "o-", color="red", label="OD (droite)", markersize=8, linewidth=1.5)

    ax.set_xscale("log")
    ax.set_xticks([250, 500, 1000, 2000, 4000, 8000])
    ax.set_xticklabels(["250", "500", "1k", "2k", "4k", "8k"])
    ax.invert_yaxis()
    ax.set_ylim(120, -10)
    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("Seuil (dB HL)")
    ax.set_title(title)
    ax.axhline(25, color="gray", linestyle=":", alpha=0.5, label="Norme (~25 dB)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_top_anomalies(
    df: pd.DataFrame,
    scores: pd.Series,
    n: int = 6,
    score_col: str = "reconstruction_error",
) -> None:
    """Trace les audiogrammes des N records les plus anormaux."""
    top_idx = scores.nlargest(n).index
    fig, axes = plt.subplots(2, n // 2, figsize=(14, 8))
    axes = axes.flatten()

    for i, idx in enumerate(top_idx):
        row = df.iloc[idx]
        score_val = scores.iloc[idx]
        plot_audiogram(
            row.get("dots_left", {}),
            row.get("dots_right", {}),
            title=f"#{idx} — score: {score_val:.3f}",
            ax=axes[i],
        )
    plt.suptitle(f"Top {n} anomalies ({score_col})", fontsize=13, fontweight="bold")
    plt.tight_layout()


# ─── Visualisation UMAP ──────────────────────────────────────────────────────

def plot_umap(
    X: np.ndarray,
    color_values: np.ndarray,
    title: str = "UMAP",
    label_name: str = "Score",
    is_categorical: bool = False,
    ax=None,
) -> None:
    """
    Projette X en 2D avec UMAP et colorie les points selon color_values.

    color_values peut être continu (scores) ou catégoriel (labels).
    """
    if not UMAP_AVAILABLE:
        print("umap-learn non installé. Lancer : pip install umap-learn")
        return

    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(X)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    if is_categorical:
        unique_vals = np.unique(color_values)
        colors = cm.tab10(np.linspace(0, 1, len(unique_vals)))
        for val, col in zip(unique_vals, colors):
            mask = color_values == val
            ax.scatter(embedding[mask, 0], embedding[mask, 1], c=[col], label=str(val), s=15, alpha=0.7)
        ax.legend(title=label_name, fontsize=8)
    else:
        sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=color_values, cmap="RdYlGn_r", s=15, alpha=0.7)
        plt.colorbar(sc, ax=ax, label=label_name)

    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()


# ─── Métriques supervisées ───────────────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, ax=None) -> None:
    """Affiche la matrice de confusion."""
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=["Normal", "Anomalie"],
        cmap="Blues",
        ax=ax,
    )


def plot_roc_curve(model, X_test: np.ndarray, y_test: np.ndarray, ax=None) -> None:
    """Trace la courbe ROC."""
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)


def plot_feature_importance(
    model,
    feature_names: list[str],
    top_n: int = 15,
    ax=None,
) -> None:
    """
    Trace l'importance des features pour Random Forest ou XGBoost.
    Aide à comprendre quelles fréquences/métriques sont discriminantes.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_values = importances[indices]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_features[::-1], top_values[::-1], color="steelblue")
    ax.set_title(f"Top {top_n} features importantes")
    ax.set_xlabel("Importance")
    plt.tight_layout()


# ─── Rapport synthétique ─────────────────────────────────────────────────────

def summary_report(df: pd.DataFrame, scores_df: pd.DataFrame) -> None:
    """Affiche un résumé texte des résultats non supervisés."""
    total = len(df)
    for col in [c for c in scores_df.columns if c.startswith("anomaly_flag")]:
        n_anomalies = scores_df[col].sum()
        print(f"{col:35s} : {n_anomalies:4d} anomalies / {total} ({100*n_anomalies/total:.1f}%)")

    if "anomaly_consensus" in scores_df.columns:
        n_consensus = scores_df["anomaly_consensus"].sum()
        print(f"\n{'anomaly_consensus (≥2/3)':35s} : {n_consensus:4d} anomalies / {total} ({100*n_consensus/total:.1f}%)")
