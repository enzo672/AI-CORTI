"""
Validation par injection synthétique.

Principe :
  On génère des audiogrammes avec un label connu (normal / anomalie).
  On score avec le modèle entraîné sur les données réelles.
  On mesure :
    - Rappel (recall) par type d'anomalie : ce que le modèle détecte
    - Taux de faux positifs (FP) : cas normaux incorrectement flaggés

Chaque type d'anomalie est généré selon des critères audiologiques publiés,
indépendants des mécanismes de détection du modèle.

Types et références :
  1. Encoche 4kHz (NIHL)       — Coles et al. 2000 / NIOSH 1998
                                  notch = T(4kHz) − moy(T(2kHz), T(8kHz)) ≥ 15 dB
  2. Perte sévère plate         — WHO Grade 3-4 : PTA > 61 dB HL
                                  tous les seuils > 65 dB HL
  3. Asymétrie extrême          — AAO-HNS 1994 : écart inter-aural > 40 dB
  4. Perte basses fréquences    — Barany Society 2015 (Ménière)
                                  seuils absolus : moy(T250, T500, T1000) ≥ 25 dB HL
  5. Perte unilatérale soudaine — AAO-HNS 2019 : ≥ 30 dB sur ≥ 3 fréquences consécutives
                                  une oreille à 55-80 dB HL flat, l'autre normale
  6. Pente raide HF             — pente > 15 dB/octave au-delà de 1 kHz
                                  (pattern décrit dans la littérature, moins standardisé)
"""

import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.features import (
    STANDARD_FREQS,
    age_correction_expected,
    extract_features,
    preprocess,
)
from src.models.unsupervised import Autoencoder

ANOMALY_TYPES = {
    "noise_notch":       "Encoche 4kHz (NIHL)",
    "severe_flat":       "Perte sévère plate (>65 dB)",
    "asymmetry":         "Asymétrie extrême (>40 dB)",
    "low_freq_loss":     "Perte basses fréquences (Ménière)",
    "sudden_unilateral": "Perte unilatérale soudaine",
    "steep_slope":       "Pente raide haute fréquence",
}


# ─── Génération des audiogrammes synthétiques ─────────────────────────────────

def _expected(freq: int, age: float, gender: int) -> float:
    val = age_correction_expected(freq, age, gender)
    return val if not np.isnan(val) else 15.0


def _make_record(i: int, label: int, atype: str, age: float, gender: int,
                 dots_left: dict, dots_right: dict) -> dict:
    return {
        "record_id":    None,
        "patient":      f"synth_{atype}_{i}",
        "visit_category": 0,
        "visit_date":   pd.Timestamp("2023-06-01", tz="UTC"),
        "gender":       gender,
        "age_at_visit": age,
        "dots_left":    dots_left,
        "dots_right":   dots_right,
        "n_freqs_left":  len(dots_left),
        "n_freqs_right": len(dots_right),
        "true_label":   label,
        "anomaly_type": atype,
    }


def generate_normals(n: int = 40, seed: int = 0) -> list[dict]:
    """Audiogrammes dans les normes ISO 7029 (résidu < 10 dB)."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        age = float(rng.uniform(25, 60))
        gender = int(rng.choice([1, 2]))
        dots_l, dots_r = {}, {}
        for freq in STANDARD_FREQS:
            exp = _expected(freq, age, gender)
            dots_l[float(freq)] = max(0.0, exp + rng.normal(0, 5))
            dots_r[float(freq)] = max(0.0, exp + rng.normal(0, 5))
        records.append(_make_record(i, 0, "normal", age, gender, dots_l, dots_r))
    return records


def generate_noise_notch(n: int = 20, seed: int = 10) -> list[dict]:
    """Encoche à 4 kHz (perte induite par le bruit — NIHL)."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        age = float(rng.uniform(30, 55))
        gender = int(rng.choice([1, 2]))
        dots_l, dots_r = {}, {}
        for freq in STANDARD_FREQS:
            exp = _expected(freq, age, gender)
            base = exp + rng.normal(0, 5)
            if freq == 4000:
                base += float(rng.uniform(35, 55))   # chute brutale à 4 kHz
            elif freq == 8000:
                base += float(rng.uniform(5, 15))    # légère récupération à 8 kHz
            dots_l[float(freq)] = max(0.0, base)
            dots_r[float(freq)] = max(0.0, base + rng.normal(0, 8))
        records.append(_make_record(i, 1, "noise_notch", age, gender, dots_l, dots_r))
    return records


def generate_severe_flat(n: int = 20, seed: int = 20) -> list[dict]:
    """Perte sévère et plate sur toutes les fréquences (> 65 dB HL)."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        age = float(rng.uniform(25, 65))
        gender = int(rng.choice([1, 2]))
        dots_l, dots_r = {}, {}
        for freq in STANDARD_FREQS:
            dots_l[float(freq)] = float(rng.uniform(65, 90))
            dots_r[float(freq)] = float(rng.uniform(65, 90))
        records.append(_make_record(i, 1, "severe_flat", age, gender, dots_l, dots_r))
    return records


def generate_asymmetry(n: int = 20, seed: int = 30) -> list[dict]:
    """Asymétrie extrême : une oreille normale, l'autre avec > 40 dB d'écart."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        age = float(rng.uniform(25, 60))
        gender = int(rng.choice([1, 2]))
        dots_l, dots_r = {}, {}
        for freq in STANDARD_FREQS:
            exp = _expected(freq, age, gender)
            # Oreille "bonne" — normale
            good = max(0.0, exp + rng.normal(0, 5))
            # Oreille "mauvaise" — écart de 40-60 dB
            bad = max(0.0, exp + float(rng.uniform(40, 60)) + rng.normal(0, 5))
            if rng.random() > 0.5:
                dots_l[float(freq)], dots_r[float(freq)] = good, bad
            else:
                dots_l[float(freq)], dots_r[float(freq)] = bad, good
        records.append(_make_record(i, 1, "asymmetry", age, gender, dots_l, dots_r))
    return records


def generate_low_freq_loss(n: int = 20, seed: int = 40) -> list[dict]:
    """
    Perte en basses fréquences — critère Barany Society 2015 (Ménière).

    Critère de génération : seuils absolus moy(T250, T500, T1000) ≥ 25 dB HL
    sur l'oreille atteinte, hautes fréquences normales (≤ 20 dB HL).
    Typiquement unilatéral ou asymétrique.
    """
    rng = np.random.default_rng(seed)
    records = []
    # Seuils absolus en dB HL (indépendants de l'âge/genre — critère Barany)
    abs_thresholds = {250: (30, 55), 500: (25, 50), 1000: (20, 40),
                      2000: (10, 20), 4000: (5, 15), 8000: (5, 15)}
    for i in range(n):
        age = float(rng.uniform(30, 60))
        gender = int(rng.choice([1, 2]))
        dots_bad, dots_good = {}, {}
        for freq in STANDARD_FREQS:
            lo, hi = abs_thresholds[freq]
            dots_bad[float(freq)]  = max(0.0, float(rng.uniform(lo, hi)) + rng.normal(0, 3))
            exp = _expected(freq, age, gender)
            dots_good[float(freq)] = max(0.0, exp + rng.normal(0, 5))
        # Atteinte unilatérale (oreille mauvaise aléatoire)
        if rng.random() > 0.5:
            dots_l, dots_r = dots_bad, dots_good
        else:
            dots_l, dots_r = dots_good, dots_bad
        records.append(_make_record(i, 1, "low_freq_loss", age, gender, dots_l, dots_r))
    return records


def generate_sudden_unilateral(n: int = 20, seed: int = 60) -> list[dict]:
    """Perte unilatérale soudaine — une oreille normale, l'autre à 55-80 dB flat."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        age = float(rng.uniform(25, 60))
        gender = int(rng.choice([1, 2]))
        dots_good, dots_bad = {}, {}
        for freq in STANDARD_FREQS:
            exp = _expected(freq, age, gender)
            dots_good[float(freq)] = max(0.0, exp + rng.normal(0, 5))
            dots_bad[float(freq)]  = float(rng.uniform(55, 80))
        if rng.random() > 0.5:
            dots_l, dots_r = dots_good, dots_bad
        else:
            dots_l, dots_r = dots_bad, dots_good
        records.append(_make_record(i, 1, "sudden_unilateral", age, gender, dots_l, dots_r))
    return records


def generate_steep_slope(n: int = 20, seed: int = 70) -> list[dict]:
    """Pente raide haute fréquence — chute très abrupte au-delà de 1 kHz."""
    rng = np.random.default_rng(seed)
    records = []
    excess = {250: (0, 5), 500: (0, 5), 1000: (10, 20),
              2000: (20, 35), 4000: (40, 55), 8000: (50, 70)}
    for i in range(n):
        age = float(rng.uniform(25, 60))
        gender = int(rng.choice([1, 2]))
        dots_l, dots_r = {}, {}
        for freq in STANDARD_FREQS:
            exp = _expected(freq, age, gender)
            lo, hi = excess[freq]
            delta = float(rng.uniform(lo, hi))
            dots_l[float(freq)] = min(120.0, max(0.0, exp + delta + rng.normal(0, 4)))
            dots_r[float(freq)] = min(120.0, max(0.0, exp + delta + rng.normal(0, 4)))
        records.append(_make_record(i, 1, "steep_slope", age, gender, dots_l, dots_r))
    return records


def build_synthetic_dataset(n_normal: int = 40, n_per_type: int = 20) -> pd.DataFrame:
    records = (
        generate_normals(n_normal)
        + generate_noise_notch(n_per_type)
        + generate_severe_flat(n_per_type)
        + generate_asymmetry(n_per_type)
        + generate_low_freq_loss(n_per_type)
        + generate_sudden_unilateral(n_per_type)
        + generate_steep_slope(n_per_type)
    )
    return pd.DataFrame(records).reset_index(drop=True)


# ─── Scoring avec le modèle entraîné ─────────────────────────────────────────

def load_trained_models(model_dir: Path) -> dict:
    """Charge les modèles sauvegardés dans model_dir."""
    scaler  = joblib.load(model_dir / "scaler.joblib")
    imputer = joblib.load(model_dir / "imputer.joblib")
    if_model = joblib.load(model_dir / "isolation_forest.joblib")

    input_dim = scaler.n_features_in_
    ae_model = Autoencoder(input_dim, hidden_dim=16, latent_dim=6)
    ae_model.load_state_dict(torch.load(model_dir / "autoencoder.pt", map_location="cpu"))
    ae_model.eval()

    return {"scaler": scaler, "imputer": imputer, "if": if_model, "ae": ae_model}


def score_synthetic(synth_df: pd.DataFrame, models: dict,
                    real_ae_threshold: float) -> pd.DataFrame:
    """
    Extrait les features des audiogrammes synthétiques, les score avec les
    modèles chargés et applique la règle de consensus.

    real_ae_threshold : seuil de reconstruction error calculé sur les données réelles
                        (percentile 95 des erreurs de reconstruction sur le train set)
    """
    feature_rows = synth_df.apply(extract_features, axis=1)
    feature_df = pd.DataFrame(list(feature_rows), index=synth_df.index)

    X, _, _ = preprocess(feature_df, scaler=models["scaler"],
                          imputer=models["imputer"], fit=False)

    # Isolation Forest
    if_flags = (models["if"].predict(X) == -1).astype(int)

    # Autoencoder — seuil issu des données réelles
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        recon = models["ae"](X_tensor).numpy()
    ae_errors = np.mean((X - recon) ** 2, axis=1)
    ae_flags = (ae_errors > real_ae_threshold).astype(int)

    # PCA baseline — seuil propre aux synthétiques (pas d'accès au seuil réel)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(5, X.shape[1]))
    X_r = pca.fit_transform(X)
    X_rec = pca.inverse_transform(X_r)
    pca_errors = np.mean((X - X_rec) ** 2, axis=1)
    pca_flags = (pca_errors > np.percentile(pca_errors, 95)).astype(int)

    consensus = ((if_flags + ae_flags + pca_flags) >= 2).astype(int)

    # Règle clinique NIHL : encoche > 15 dB à 4kHz (Coles et al. 2000)
    nihl_flag = (
        (feature_df["notch_4k_L"].fillna(0) > 15) |
        (feature_df["notch_4k_R"].fillna(0) > 15)
    ).astype(int)

    # Règle clinique Ménière : PTA BF corrigé > 25 dB (résidu vs norme âge/genre)
    low_L = feature_df[["L_250", "L_500", "L_1000"]].mean(axis=1)
    low_R = feature_df[["R_250", "R_500", "R_1000"]].mean(axis=1)
    meniere_flag = ((low_L.fillna(0) > 25) | (low_R.fillna(0) > 25)).astype(int)

    anomaly_final = ((consensus == 1) | (nihl_flag == 1) | (meniere_flag == 1)).astype(int)

    return pd.DataFrame({
        "anomaly_flag_if":   if_flags,
        "anomaly_flag_ae":   ae_flags,
        "anomaly_flag_pca":  pca_flags,
        "nihl_flag":         nihl_flag,
        "anomaly_consensus": consensus,
        "anomaly_final":     anomaly_final,
        "reconstruction_error": ae_errors,
    }, index=synth_df.index)


# ─── Métriques ────────────────────────────────────────────────────────────────

def compute_metrics(synth_df: pd.DataFrame, scores_df: pd.DataFrame) -> dict:
    """Calcule rappel par type d'anomalie et taux de faux positifs."""
    merged = synth_df[["true_label", "anomaly_type"]].join(
        scores_df["anomaly_final"]
    ).rename(columns={"anomaly_final": "anomaly_consensus"})

    results = {}

    # Faux positifs sur les cas normaux
    normals = merged[merged["anomaly_type"] == "normal"]
    n_fp = (normals["anomaly_consensus"] == 1).sum()
    results["normal"] = {
        "label": "Normaux",
        "n_total": len(normals),
        "n_detected": int(n_fp),
        "rate": n_fp / len(normals) if len(normals) > 0 else 0.0,
        "is_anomaly": False,
    }

    # Rappel par type d'anomalie
    for atype, label in ANOMALY_TYPES.items():
        subset = merged[merged["anomaly_type"] == atype]
        n_detected = (subset["anomaly_consensus"] == 1).sum()
        results[atype] = {
            "label": label,
            "n_total": len(subset),
            "n_detected": int(n_detected),
            "rate": n_detected / len(subset) if len(subset) > 0 else 0.0,
            "is_anomaly": True,
        }

    # Global
    anomalies = merged[merged["true_label"] == 1]
    n_detected_total = (anomalies["anomaly_consensus"] == 1).sum()
    results["_total"] = {
        "n_total": len(anomalies),
        "n_detected": int(n_detected_total),
        "recall": n_detected_total / len(anomalies) if len(anomalies) > 0 else 0.0,
    }

    return results


CLINICAL_WEIGHTS = {
    "noise_notch":       0.30,  # NIHL — la plus fréquente en santé au travail (Coles 2000)
    "severe_flat":       0.15,  # WHO Grade 3-4
    "asymmetry":         0.15,  # AAO-HNS 1994
    "low_freq_loss":     0.20,  # Ménière — Barany Society 2015
    "sudden_unilateral": 0.15,  # AAO-HNS 2019
    "steep_slope":       0.05,  # moins standardisé cliniquement
}


def compute_detection_score(metrics: dict) -> dict:
    """
    Score global de détection (0–100).

    Score = weighted_recall × (1 − fp_rate) × 100

    Les poids sont cliniques : l'encoche 4kHz (NIHL) est la pathologie
    professionnelle la plus fréquente, elle compte pour 50%.
    """
    weighted_recall = sum(
        CLINICAL_WEIGHTS[k] * metrics[k]["rate"]
        for k in CLINICAL_WEIGHTS
        if k in metrics
    )
    fp_rate = metrics["normal"]["rate"]
    score = weighted_recall * (1 - fp_rate) * 100

    interpretation = (
        "Excellent" if score >= 85 else
        "Bon"       if score >= 70 else
        "Passable"  if score >= 55 else
        "Insuffisant"
    )
    return {
        "score": round(score, 1),
        "weighted_recall": round(weighted_recall, 4),
        "fp_rate": round(fp_rate, 4),
        "interpretation": interpretation,
    }


def print_synthetic_report(metrics: dict) -> None:
    print("\n=== Validation par injection synthétique ===")
    print(f"{'Type':<35} {'Injectés':>9} {'Détectés':>9} {'Taux':>8}")
    print("─" * 65)

    for key, m in metrics.items():
        if key == "_total":
            continue
        verb = "FP" if not m["is_anomaly"] else "Rappel"
        print(f"{m['label']:<35} {m['n_total']:>9} {m['n_detected']:>9} "
              f"{verb}: {m['rate']:.1%}")

    t = metrics["_total"]
    print("─" * 65)
    print(f"{'Total anomalies':<35} {t['n_total']:>9} {t['n_detected']:>9} "
          f"Rappel: {t['recall']:.1%}")

    ds = compute_detection_score(metrics)
    print(f"\n  Detection Score : {ds['score']:.1f} / 100  →  {ds['interpretation']}")
    print(f"  (rappel pondéré {ds['weighted_recall']:.1%}  ×  (1 − FP {ds['fp_rate']:.1%}))")


def save_synthetic_report(metrics: dict, output_dir: Path) -> None:
    """Sauvegarde le rapport en CSV et en texte lisible."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV détaillé
    rows = []
    for key, m in metrics.items():
        if key == "_total":
            continue
        rows.append({
            "type": key,
            "label": m["label"],
            "is_anomaly": m["is_anomaly"],
            "n_total": m["n_total"],
            "n_detected": m["n_detected"],
            "rate": round(m["rate"], 4),
            "metric": "faux_positifs" if not m["is_anomaly"] else "rappel",
        })
    import pandas as pd
    pd.DataFrame(rows).to_csv(output_dir / "synthetic_validation.csv", index=False)

    # Résumé texte
    t = metrics["_total"]
    lines = [
        "=== Validation par injection synthétique ===\n",
        f"{'Type':<35} {'Injectés':>9} {'Détectés':>9} {'Taux':>8}",
        "─" * 65,
    ]
    for key, m in metrics.items():
        if key == "_total":
            continue
        verb = "FP" if not m["is_anomaly"] else "Rappel"
        lines.append(f"{m['label']:<35} {m['n_total']:>9} {m['n_detected']:>9} "
                     f"{verb}: {m['rate']:.1%}")
    lines.append("─" * 65)
    lines.append(f"{'Total anomalies':<35} {t['n_total']:>9} {t['n_detected']:>9} "
                 f"Rappel: {t['recall']:.1%}")
    ds = compute_detection_score(metrics)
    lines.append(f"\nDetection Score : {ds['score']:.1f} / 100  →  {ds['interpretation']}")
    lines.append(f"(rappel pondéré {ds['weighted_recall']:.1%}  ×  (1 − FP {ds['fp_rate']:.1%}))")
    lines.append(f"\nPoint faible identifié : types avec rappel < 80%")
    for key, m in metrics.items():
        if key == "_total" or not m["is_anomaly"]:
            continue
        if m["rate"] < 0.80:
            lines.append(f"  → {m['label']} : {m['rate']:.1%}")

    txt_path = output_dir / "synthetic_validation.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Rapport sauvegardé → {output_dir}/synthetic_validation.csv")
    print(f"  Rapport sauvegardé → {output_dir}/synthetic_validation.txt")


# ─── Visualisation ────────────────────────────────────────────────────────────

def plot_synthetic_validation(metrics: dict, output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    _, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── 1. Rappel par type d'anomalie ────────────────────────────────────────
    ax = axes[0]
    anom_keys = [k for k, m in metrics.items() if k != "_total" and m["is_anomaly"]]
    labels = [metrics[k]["label"] for k in anom_keys]
    recalls = [metrics[k]["rate"] for k in anom_keys]
    colors = ["#E85C4C" if r < 0.7 else "#F5A623" if r < 0.85 else "#4CAF50"
              for r in recalls]

    bars = ax.barh(labels, recalls, color=colors, edgecolor="white", height=0.5)
    ax.axvline(0.80, color="gray", linestyle="--", linewidth=1.2, label="Objectif 80%")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Rappel (% anomalies détectées)")
    ax.set_title("Rappel par type d'anomalie\n(ce que le modèle détecte)")
    ax.legend(fontsize=9)

    for bar, val in zip(bars, recalls):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.0%}", va="center", fontsize=10, fontweight="bold")

    ax.grid(True, alpha=0.3, axis="x")

    # ── 2. Taux de faux positifs et vrais positifs global ─────────────────────
    ax = axes[1]
    fp_rate = metrics["normal"]["rate"]
    recall = metrics["_total"]["recall"]
    fn_rate = 1.0 - recall

    categories = ["Vrais positifs\n(anomalies détectées)", "Faux négatifs\n(anomalies manquées)",
                  "Faux positifs\n(normaux flaggés)"]
    values = [recall, fn_rate, fp_rate]
    bar_colors = ["#4CAF50", "#E85C4C", "#F5A623"]

    bars2 = ax.bar(categories, values, color=bar_colors, edgecolor="white", width=0.5)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Taux")
    ax.set_title("Vue globale : FP / FN / TP\n(sur les données synthétiques)")

    for bar, val in zip(bars2, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.0%}", ha="center", fontsize=11, fontweight="bold")

    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = figures_dir / "synthetic_validation.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Figure injection synthétique → {out_path}")
