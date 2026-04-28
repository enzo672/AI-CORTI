"""
Validation de la détection d'anomalies par rapport à la norme ISO 7029 / OSHA Appendix F.

Principe :
  Pour chaque audiogramme, on calcule le résidu normé (mesuré − attendu pour cet âge/genre).
  Si la méthode consensus est efficace, les audiogrammes flaggés doivent avoir des résidus
  significativement plus élevés que les audiogrammes non flaggés.

Métrique principale : Precision@ISO7029
  Parmi les N records flaggés anomalie, combien dépassent un seuil de résidu (ex. 15 dB) ?
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.features import (
    STANDARD_FREQS,
    interpolate_thresholds,
    apply_age_correction,
)

RESIDUAL_THRESHOLD_DB = 15.0  # dB au-dessus de la norme = déviation cliniquement significative


# ─── Calcul des résidus ───────────────────────────────────────────────────────

def compute_iso7029_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule, pour chaque record, la déviation maximale par rapport à la norme ISO 7029.

    Colonnes retournées :
      iso7029_max_residual   : max du résidu sur toutes les fréquences et les deux oreilles
      iso7029_mean_residual  : moyenne des résidus positifs (excès au-dessus de la norme)
      iso7029_has_age_gender : True si age et gender étaient disponibles pour le calcul
    """
    rows = []
    for _, row in df.iterrows():
        age = row.get("age_at_visit")
        gender = row.get("gender")

        age_valid = age is not None and not (isinstance(age, float) and np.isnan(age))
        gender_valid = (
            gender is not None
            and not (isinstance(gender, float) and np.isnan(gender))
            and int(gender) in (1, 2)
        )

        if not (age_valid and gender_valid):
            rows.append({
                "iso7029_max_residual": np.nan,
                "iso7029_mean_residual": np.nan,
                "iso7029_has_age_gender": False,
            })
            continue

        left_raw = interpolate_thresholds(row["dots_left"], STANDARD_FREQS)
        right_raw = interpolate_thresholds(row["dots_right"], STANDARD_FREQS)

        left_res = apply_age_correction(left_raw, float(age), int(gender))
        right_res = apply_age_correction(right_raw, float(age), int(gender))

        all_residuals = [
            v for v in list(left_res.values()) + list(right_res.values())
            if not np.isnan(v)
        ]

        if not all_residuals:
            rows.append({
                "iso7029_max_residual": np.nan,
                "iso7029_mean_residual": np.nan,
                "iso7029_has_age_gender": True,
            })
            continue

        positive_residuals = [v for v in all_residuals if v > 0]

        rows.append({
            "iso7029_max_residual": float(np.max(all_residuals)),
            "iso7029_mean_residual": float(np.mean(positive_residuals)) if positive_residuals else 0.0,
            "iso7029_has_age_gender": True,
        })

    return pd.DataFrame(rows, index=df.index)


# ─── Métriques de validation ──────────────────────────────────────────────────

def compute_precision_at_iso7029(
    residuals_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    threshold_db: float = RESIDUAL_THRESHOLD_DB,
) -> dict:
    """
    Calcule la Precision@ISO7029 : parmi les records flaggés anomalie (consensus),
    quelle proportion dépasse le seuil de résidu normé ?

    Retourne un dict avec les métriques clés.
    """
    merged = residuals_df.join(scores_df[["anomaly_consensus"]])
    valid = merged[merged["iso7029_has_age_gender"] & merged["iso7029_max_residual"].notna()]

    if valid.empty:
        return {"error": "Aucun record avec age/genre disponible"}

    flagged = valid[valid["anomaly_consensus"] == 1]
    normal = valid[valid["anomaly_consensus"] == 0]

    n_flagged = len(flagged)
    n_flagged_above = (flagged["iso7029_max_residual"] > threshold_db).sum() if n_flagged > 0 else 0
    precision = n_flagged_above / n_flagged if n_flagged > 0 else np.nan

    mean_flagged = flagged["iso7029_max_residual"].mean()
    mean_normal = normal["iso7029_max_residual"].mean()

    return {
        "n_valid_records": len(valid),
        "n_flagged": n_flagged,
        "n_flagged_above_threshold": int(n_flagged_above),
        "threshold_db": threshold_db,
        "precision_at_iso7029": round(float(precision), 3) if not np.isnan(precision) else None,
        "mean_residual_flagged_dB": round(float(mean_flagged), 2) if not np.isnan(mean_flagged) else None,
        "mean_residual_normal_dB": round(float(mean_normal), 2) if not np.isnan(mean_normal) else None,
    }


def print_iso7029_report(metrics: dict) -> None:
    print("\n=== Validation ISO 7029 ===")
    if "error" in metrics:
        print(f"  {metrics['error']}")
        return

    print(f"  Records avec âge/genre       : {metrics['n_valid_records']}")
    print(f"  Records flaggés anomalie      : {metrics['n_flagged']}")
    print(f"  Flaggés dépassant {metrics['threshold_db']:.0f} dB résidu : "
          f"{metrics['n_flagged_above_threshold']} "
          f"→ Precision@ISO7029 = {metrics['precision_at_iso7029']:.1%}" if metrics['precision_at_iso7029'] is not None else "  N/A")
    print(f"  Résidu moyen — anomalies      : {metrics['mean_residual_flagged_dB']:.1f} dB" if metrics['mean_residual_flagged_dB'] is not None else "")
    print(f"  Résidu moyen — normaux        : {metrics['mean_residual_normal_dB']:.1f} dB" if metrics['mean_residual_normal_dB'] is not None else "")


# ─── Visualisation ────────────────────────────────────────────────────────────

def plot_iso7029_validation(
    residuals_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    output_dir,
    threshold_db: float = RESIDUAL_THRESHOLD_DB,
) -> None:
    """
    Génère deux graphiques de validation ISO 7029 :
      1. Box plot : résidu maximal — anomalies vs normaux
      2. Scatter  : résidu ISO 7029 vs erreur de reconstruction (AE)
    """
    from pathlib import Path
    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    merged = residuals_df.join(scores_df[["anomaly_consensus", "reconstruction_error"]])
    valid = merged[merged["iso7029_has_age_gender"] & merged["iso7029_max_residual"].notna()].copy()

    if valid.empty:
        print("  Validation ISO 7029 ignorée : aucun record avec âge/genre.")
        return

    valid["Groupe"] = valid["anomaly_consensus"].map({0: "Normal", 1: "Anomalie"})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── 1. Box plot ───────────────────────────────────────────────────────────
    ax = axes[0]
    groups = [
        valid.loc[valid["anomaly_consensus"] == 0, "iso7029_max_residual"].dropna(),
        valid.loc[valid["anomaly_consensus"] == 1, "iso7029_max_residual"].dropna(),
    ]
    bp = ax.boxplot(groups, labels=["Normal", "Anomalie (consensus)"],
                    patch_artist=True, widths=0.5)
    colors = ["#4C9BE8", "#E85C4C"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(threshold_db, color="orange", linestyle="--", linewidth=1.5,
               label=f"Seuil {threshold_db:.0f} dB (clinique)")
    ax.set_ylabel("Résidu maximal ISO 7029 (dB)")
    ax.set_title("Déviation vs norme ISO 7029\npar groupe de détection")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # ── 2. Scatter résidu vs reconstruction_error ─────────────────────────────
    ax = axes[1]
    for label, color, marker in [("Normal", "#4C9BE8", "o"), ("Anomalie", "#E85C4C", "^")]:
        subset = valid[valid["Groupe"] == label]
        ax.scatter(
            subset["reconstruction_error"],
            subset["iso7029_max_residual"],
            c=color, marker=marker, alpha=0.6, s=30, label=label,
        )

    ax.axhline(threshold_db, color="orange", linestyle="--", linewidth=1.2,
               label=f"Seuil ISO 7029 ({threshold_db:.0f} dB)")
    ax.set_xlabel("Erreur de reconstruction (Autoencoder)")
    ax.set_ylabel("Résidu maximal ISO 7029 (dB)")
    ax.set_title("Corrélation : AE vs ISO 7029")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = figures_dir / "iso7029_validation.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Figure ISO 7029 sauvegardée → {out_path}")
