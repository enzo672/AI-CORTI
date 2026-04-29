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
    _, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)

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
    _, axes = plt.subplots(2, n // 2, figsize=(14, 8))
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
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
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


# ─── Vérification sans vérité terrain ────────────────────────────────────────

def plot_flag_overlap(scores_df: pd.DataFrame) -> None:
    """
    Matrices de confusion ML (consensus) vs chaque règle clinique.

    Les cas flaggés par les deux mécanismes indépendants sont les plus fiables.
    """
    pairs = [
        ("anomaly_consensus", "nihl_flag",    "NIHL"),
        ("anomaly_consensus", "meniere_flag",  "Ménière"),
        ("anomaly_consensus", "sts_flag",      "STS OSHA"),
    ]
    available = [(a, b, lbl) for a, b, lbl in pairs
                 if a in scores_df.columns and b in scores_df.columns]
    if not available:
        print("Colonnes de flags manquantes dans scores_df.")
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (col_ml, col_rule, label) in zip(axes, available):
        ml   = scores_df[col_ml].fillna(0).astype(int)
        rule = scores_df[col_rule].fillna(0).astype(int)
        matrix = pd.crosstab(
            ml.map({0: "ML: normal", 1: "ML: anomalie"}),
            rule.map({0: f"{label}: normal", 1: f"{label}: flag"}),
        )
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax,
                    linewidths=0.5, cbar=False)
        ax.set_title(f"Consensus ML × {label}")
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.suptitle("Accord ML vs règles cliniques — cases diagonales = consensus",
                 fontsize=11, y=1.02)
    plt.tight_layout()


def plot_rule_distributions(
    feature_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    delta_df: pd.DataFrame | None = None,
) -> None:
    """
    Distribution de la métrique sous-jacente à chaque règle clinique.

    Un seuil bien calibré coupe une vraie bimodalité, pas une zone de bruit.
    """
    from src.features import STANDARD_FREQS

    panels = []

    notch_cols_L = [c for c in ["notch_3k_L", "notch_4k_L", "notch_6k_L"] if c in feature_df.columns]
    notch_cols_R = [c for c in ["notch_3k_R", "notch_4k_R", "notch_6k_R"] if c in feature_df.columns]
    if notch_cols_L and notch_cols_R:
        notch_max = pd.concat([
            feature_df[notch_cols_L].max(axis=1),
            feature_df[notch_cols_R].max(axis=1),
        ], axis=1).max(axis=1)
        panels.append((notch_max, 15, "Encoche max (3/4/6 kHz) [dB]", "NIHL > 15 dB"))

    if "low_freq_pta_L" in feature_df.columns and "low_freq_pta_R" in feature_df.columns:
        lf_max = feature_df[["low_freq_pta_L", "low_freq_pta_R"]].max(axis=1)
        panels.append((lf_max, 25, "PTA basses fréquences max [dB]", "Ménière > 25 dB"))

    if delta_df is not None and "sts_L" in delta_df.columns and "sts_R" in delta_df.columns:
        sts_max = delta_df[["sts_L", "sts_R"]].max(axis=1).dropna()
        panels.append((sts_max, 10, "STS OSHA max [dB]", "STS ≥ 10 dB"))

    if not panels:
        print("Aucune métrique disponible pour plot_rule_distributions.")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]

    for ax, (values, threshold, xlabel, label) in zip(axes, panels):
        v = values.dropna()
        ax.hist(v, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Seuil : {label}")
        pct = 100 * (v >= threshold).mean()
        ax.set_title(f"{pct:.1f} % flaggés")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Nombre d'audiogrammes")
        ax.legend(fontsize=8)

    plt.suptitle("Distribution des métriques cliniques", fontsize=12)
    plt.tight_layout()


def plot_nihl_mean_profile(
    feature_df: pd.DataFrame,
    scores_df: pd.DataFrame,
) -> None:
    """
    Profil audiométrique moyen des audiogrammes flaggés NIHL vs non-flaggés.

    Si la règle est correcte, les flaggés doivent montrer une encoche visible
    à 3–6 kHz sur le profil moyen (zone ombrée = ±1 écart-type).
    """
    from src.features import STANDARD_FREQS

    freqs = STANDARD_FREQS
    nihl = scores_df["nihl_flag"].fillna(0).astype(bool)
    nihl.index = feature_df.index

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, ear, label in zip(axes, ["L", "R"], ["Gauche (OG)", "Droite (OD)"]):
        cols = [f"{ear}_{f}" for f in freqs if f"{ear}_{f}" in feature_df.columns]
        f_vals = [f for f in freqs if f"{ear}_{f}" in feature_df.columns]
        if not cols:
            continue

        for flag_val, color, name in [(True, "tomato", "NIHL flaggé"), (False, "steelblue", "Non flaggé")]:
            subset = feature_df.loc[nihl == flag_val, cols]
            if subset.empty:
                continue
            mean = subset.mean()
            std  = subset.std()
            ax.plot(f_vals, mean.values, color=color, linewidth=2, label=f"{name} (n={len(subset)})")
            ax.fill_between(f_vals, mean - std, mean + std, alpha=0.15, color=color)

        ax.set_xscale("log")
        ax.set_xticks(freqs)
        ax.set_xticklabels([str(f) if f < 1000 else f"{f//1000}k" for f in freqs])
        ax.invert_yaxis()
        ax.set_ylim(80, -15)
        ax.set_xlabel("Fréquence (Hz)")
        ax.set_ylabel("Résidu corrigé moyen (dB)")
        ax.set_title(f"Oreille {label}")
        ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

    plt.suptitle("Profil moyen NIHL flaggé vs non-flaggé (±1 σ)", fontsize=12)
    plt.tight_layout()


def plot_prevalence_check(scores_df: pd.DataFrame) -> None:
    """
    Compare les taux de flags observés aux plages épidémiologiques attendues.

    Un taux hors plage indique un seuil probablement mal calibré.
    """
    # (colonne, label, borne_basse_%, borne_haute_%, couleur_barre)
    rules = [
        ("nihl_flag",        "NIHL",         15, 30, "steelblue"),
        ("meniere_flag",     "Ménière",        1,  5, "mediumseagreen"),
        ("sts_flag",         "STS OSHA",       5, 15, "darkorange"),
        ("anomaly_consensus","Consensus ML", None, None, "slategray"),
    ]

    n = len(scores_df)
    labels, rates, lo, hi, colors = [], [], [], [], []
    for col, label, lo_pct, hi_pct, color in rules:
        if col not in scores_df.columns:
            continue
        rate = 100 * scores_df[col].fillna(0).mean()
        labels.append(label)
        rates.append(rate)
        lo.append(lo_pct)
        hi.append(hi_pct)
        colors.append(color)

    _, ax = plt.subplots(figsize=(8, 4))
    x = range(len(labels))
    bars = ax.bar(x, rates, color=colors, alpha=0.85, edgecolor="white", width=0.5)

    for i, (l, h) in enumerate(zip(lo, hi)):
        if l is not None:
            ax.axhspan(l, h, xmin=(i) / len(labels), xmax=(i + 1) / len(labels),
                       color="gold", alpha=0.25, linewidth=0)
            ax.text(i, h + 0.5, f"attendu {l}–{h}%", ha="center", va="bottom",
                    fontsize=7.5, color="goldenrod")

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("% audiogrammes flaggés")
    ax.set_title(f"Prévalence des flags (n={n}) vs plages épidémiologiques attendues")
    ax.set_ylim(0, max(rates + [31]) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()


def plot_young_baseline_fpr(df: pd.DataFrame, scores_df: pd.DataFrame) -> None:
    """
    Taux de flags anomaly_final par tranche d'âge sur les Baseline.

    Les Baseline de sujets jeunes (< 30 ans) devraient être quasi-normaux.
    Un taux élevé chez les jeunes signale des faux positifs.
    """
    if "anomaly_final" not in scores_df.columns:
        print("Colonne anomaly_final absente de scores_df.")
        return

    joined = df[["age_at_visit", "visit_category"]].copy()
    joined.index = scores_df.index
    joined["anomaly_final"] = scores_df["anomaly_final"].values

    bins   = [0, 30, 45, 60, 120]
    labels = ["< 30", "30–45", "45–60", "> 60"]
    joined["age_bin"] = pd.cut(joined["age_at_visit"], bins=bins, labels=labels, right=False)

    visit_labels = {0: "Baseline", 1: "Periodic", 2: "Depart"}
    joined["visit_label"] = joined["visit_category"].map(visit_labels).fillna("?")

    rates = (
        joined.groupby(["age_bin", "visit_label"], observed=True)["anomaly_final"]
        .mean() * 100
    ).reset_index()
    rates.columns = ["age_bin", "visit_label", "flag_rate"]

    pivot = rates.pivot(index="age_bin", columns="visit_label", values="flag_rate").fillna(0)

    _, ax = plt.subplots(figsize=(9, 4))
    pivot.plot(kind="bar", ax=ax, edgecolor="white", alpha=0.85)

    ax.axhline(5, color="red", linestyle="--", linewidth=1, alpha=0.6, label="Seuil FP indicatif 5%")
    ax.set_xlabel("Tranche d'âge")
    ax.set_ylabel("% anomaly_final flaggés")
    ax.set_title("Taux de flags par âge et type de visite")
    ax.legend(title="Type de visite", fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(axis="y", alpha=0.3)
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
