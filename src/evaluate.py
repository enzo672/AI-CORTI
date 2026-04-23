"""
Visualisation et évaluation des résultats de détection d'anomalies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

VISIT_LABELS = {0: "Baseline", 1: "Periodic", 2: "Depart"}


# ─── Audiogramme ─────────────────────────────────────────────────────────────

def plot_audiogram(
    dots_left: dict,
    dots_right: dict,
    title: str = "Audiogramme",
    ax=None,
) -> None:
    """
    Trace un audiogramme standard : OG (×, bleu) et OD (○, rouge).
    Axe Y inversé — convention audiométrique (0 dB en haut).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    if dots_left:
        freqs_l = sorted(dots_left.keys())
        ax.plot(freqs_l, [dots_left[f] for f in freqs_l],
                "x-", color="blue", label="OG (gauche)", markersize=10, linewidth=1.5)

    if dots_right:
        freqs_r = sorted(dots_right.keys())
        ax.plot(freqs_r, [dots_right[f] for f in freqs_r],
                "o-", color="red", label="OD (droite)", markersize=8, linewidth=1.5)

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


def plot_patient_trajectory(
    df: pd.DataFrame,
    patient: str,
    score_col: str | None = None,
    scores_df: pd.DataFrame | None = None,
) -> None:
    """
    Trace l'évolution temporelle d'un patient : Baseline → Periodic(s) → Depart.

    patient    : hash patient (colonne df["patient"])
    score_col  : colonne de scores dans scores_df à afficher dans le titre
    scores_df  : DataFrame de scores aligné sur df
    """
    patient_df = df[df["patient"] == patient].sort_values("visit_date")

    if patient_df.empty:
        print(f"Aucun record pour le patient {patient[:16]}...")
        return

    n = len(patient_df)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)

    for i, (idx, row) in enumerate(patient_df.iterrows()):
        label = VISIT_LABELS.get(row.get("visit_category"), "?")
        date_str = ""
        if pd.notna(row.get("visit_date")):
            date_str = f" ({row['visit_date'].date()})"

        score_str = ""
        if score_col is not None and scores_df is not None and idx in scores_df.index:
            val = scores_df.loc[idx, score_col]
            if not np.isnan(val):
                score_str = f"\n{score_col}: {val:.3f}"

        plot_audiogram(
            row["dots_left"],
            row["dots_right"],
            title=f"{label}{date_str}{score_str}",
            ax=axes[0, i],
        )

    plt.suptitle(f"Trajectoire patient {patient[:16]}...", fontsize=12, fontweight="bold")
    plt.tight_layout()


# ─── Scores d'anomalie ────────────────────────────────────────────────────────

def plot_anomaly_score_distribution(
    scores: pd.Series,
    threshold: float | None = None,
    title: str = "Distribution des scores d'anomalie",
    ax=None,
) -> None:
    """Histogramme des scores avec seuil optionnel."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores.dropna(), bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    if threshold is not None:
        ax.axvline(threshold, color="red", linestyle="--", label=f"Seuil = {threshold:.3f}")
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Score")
    ax.set_ylabel("Nombre de samples")
    plt.tight_layout()


def plot_top_anomalies(
    df: pd.DataFrame,
    scores: pd.Series,
    n: int = 6,
    score_col: str = "reconstruction_error",
) -> None:
    """Trace les audiogrammes des N records les plus anormaux."""
    top_idx = scores.dropna().nlargest(n).index
    fig, axes = plt.subplots(2, n // 2, figsize=(14, 8))
    axes = axes.flatten()

    for i, idx in enumerate(top_idx):
        row = df.loc[idx]
        label = VISIT_LABELS.get(row.get("visit_category"), "?")
        plot_audiogram(
            row.get("dots_left", {}),
            row.get("dots_right", {}),
            title=f"#{idx} [{label}] — {scores.loc[idx]:.3f}",
            ax=axes[i],
        )
    plt.suptitle(f"Top {n} anomalies ({score_col})", fontsize=13, fontweight="bold")
    plt.tight_layout()


# ─── Delta / évolution ────────────────────────────────────────────────────────

def plot_delta_heatmap(
    delta_df: pd.DataFrame,
    df: pd.DataFrame | None = None,
    ear: str = "L",
) -> None:
    """
    Heatmap des deltas fréquentiels pour tous les records Periodic/Depart.

    Chaque ligne = un record, chaque colonne = une fréquence standard.
    Rouge = aggravation (hausse du seuil), Bleu = amélioration.
    """
    from src.features import STANDARD_FREQS
    cols = [f"delta_{ear}_{f}" for f in STANDARD_FREQS]
    available = [c for c in cols if c in delta_df.columns]

    if not available:
        print(f"Pas de colonnes delta_{ear}_* dans le DataFrame.")
        return

    data = delta_df[available].dropna(how="all")

    if df is not None:
        labels = df.loc[data.index, "visit_category"].map(VISIT_LABELS).fillna("?")
    else:
        labels = data.index.astype(str)

    _, ax = plt.subplots(figsize=(10, max(4, len(data) * 0.4)))
    sns.heatmap(
        data,
        cmap="RdBu_r",
        center=0,
        ax=ax,
        xticklabels=[str(f) for f in STANDARD_FREQS[:len(available)]],
        yticklabels=labels.tolist(),
        cbar_kws={"label": "Δ dB vs Baseline (+ = aggravation)"},
    )
    ax.set_title(f"Delta vs Baseline — oreille {'gauche' if ear == 'L' else 'droite'}")
    ax.set_xlabel("Fréquence (Hz)")
    plt.tight_layout()


def plot_sts_distribution(delta_df: pd.DataFrame) -> None:
    """Distribution des STS (Standard Threshold Shift) OG et OD."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    threshold = 10.0

    for ax, col, label in zip(axes, ["sts_L", "sts_R"], ["OG (gauche)", "OD (droite)"]):
        if col not in delta_df.columns:
            continue
        values = delta_df[col].dropna()
        ax.hist(values, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(threshold, color="red", linestyle="--", label=f"STS ≥ {threshold} dB")
        ax.axvline(0, color="gray", linestyle=":", alpha=0.7)
        n_sts = (values >= threshold).sum()
        ax.set_title(f"STS {label} — {n_sts} cas ≥ {threshold} dB ({100*n_sts/len(values):.1f}%)")
        ax.set_xlabel("Shift moyen 2k/3k/4k Hz (dB)")
        ax.set_ylabel("Nombre de records")
        ax.legend()

    plt.suptitle("Standard Threshold Shift (OSHA)", fontsize=12, fontweight="bold")
    plt.tight_layout()


# ─── UMAP ────────────────────────────────────────────────────────────────────

def plot_umap(
    X: np.ndarray,
    color_values: np.ndarray,
    title: str = "UMAP",
    label_name: str = "Score",
    is_categorical: bool = False,
    ax=None,
) -> None:
    """Projette X en 2D avec UMAP et colorie les points selon color_values."""
    if not UMAP_AVAILABLE:
        print("umap-learn non installé. Lancer : pip install umap-learn")
        return

    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(X)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    if is_categorical:
        unique_vals = np.unique(color_values[~pd.isna(color_values)])
        colors = cm.tab10(np.linspace(0, 1, len(unique_vals)))
        for val, col in zip(unique_vals, colors):
            mask = color_values == val
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=[col], label=str(val), s=15, alpha=0.7)
        ax.legend(title=label_name, fontsize=8)
    else:
        sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                        c=color_values, cmap="RdYlGn_r", s=15, alpha=0.7)
        plt.colorbar(sc, ax=ax, label=label_name)

    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()


# ─── Rapport synthétique ─────────────────────────────────────────────────────

def summary_report(df: pd.DataFrame, scores_df: pd.DataFrame) -> None:
    """Affiche un résumé texte des résultats non supervisés."""
    total = len(df)
    for col in [c for c in scores_df.columns if c.startswith("anomaly_flag")]:
        n_anomalies = scores_df[col].sum()
        print(f"{col:35s} : {n_anomalies:4.0f} / {total} ({100*n_anomalies/total:.1f}%)")

    if "anomaly_consensus" in scores_df.columns:
        n_consensus = scores_df["anomaly_consensus"].sum()
        print(f"\n{'anomaly_consensus (≥2/3)':35s} : {n_consensus:4.0f} / {total} ({100*n_consensus/total:.1f}%)")
