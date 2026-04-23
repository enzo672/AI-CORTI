"""
Feature engineering pour les audiogrammes.

Deux types de features :
- Absolues  : seuils dB interpolés + dérivées (PTA, asymétrie, chute HF)
- Delta     : changement vs Baseline du même patient (Periodic / Depart)
              → plus robuste cliniquement car normalise la variabilité inter-individuelle
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

STANDARD_FREQS = [250, 500, 1000, 2000, 4000, 8000]

PTA_FREQS = [500, 1000, 2000, 4000]

# Standard Threshold Shift (OSHA) : moyenne à 2000, 3000, 4000 Hz
# ≥ 10 dB = shift cliniquement significatif
STS_FREQS = [2000, 3000, 4000]

VISIT_CATEGORY_LABELS = {0: "Baseline", 1: "Periodic", 2: "Depart"}


# ─── Fonctions de base ────────────────────────────────────────────────────────

def interpolate_thresholds(dots: dict, target_freqs: list[int]) -> dict:
    """
    Interpole les seuils auditifs aux fréquences cibles.

    Si une fréquence cible est hors de la plage des fréquences mesurées,
    retourne NaN (pas d'extrapolation).
    """
    if not dots:
        return {f: np.nan for f in target_freqs}

    freqs = sorted(dots.keys())
    dbs = [dots[f] for f in freqs]

    if len(freqs) == 1:
        return {tf: (dbs[0] if tf == freqs[0] else np.nan) for tf in target_freqs}

    interpolator = interp1d(freqs, dbs, kind="linear", bounds_error=False, fill_value=np.nan)
    return {tf: float(interpolator(tf)) for tf in target_freqs}


def compute_pta(thresholds: dict, pta_freqs: list[int] = PTA_FREQS) -> float:
    """Pure Tone Average : moyenne des seuils aux fréquences de référence."""
    values = [thresholds[f] for f in pta_freqs if not np.isnan(thresholds.get(f, np.nan))]
    return float(np.mean(values)) if values else np.nan


def compute_high_freq_drop(thresholds: dict) -> float:
    """Chute haute fréquence : différence 8000 Hz − 4000 Hz."""
    v4 = thresholds.get(4000, np.nan)
    v8 = thresholds.get(8000, np.nan)
    if np.isnan(v4) or np.isnan(v8):
        return np.nan
    return float(v8 - v4)


def compute_asymmetry(left_thresh: dict, right_thresh: dict) -> float:
    """Asymétrie inter-oreilles : moyenne de |OG − OD| sur les fréquences communes."""
    diffs = []
    for f in STANDARD_FREQS:
        l_val = left_thresh.get(f, np.nan)
        r_val = right_thresh.get(f, np.nan)
        if not np.isnan(l_val) and not np.isnan(r_val):
            diffs.append(abs(l_val - r_val))
    return float(np.mean(diffs)) if diffs else np.nan


def compute_sts(current_dots: dict, baseline_dots: dict) -> float:
    """
    Standard Threshold Shift (OSHA) : shift moyen aux fréquences 2000/3000/4000 Hz.

    Valeur ≥ 10 dB = détérioration cliniquement significative vs Baseline.
    Interpolation à 3000 Hz si non mesuré directement.
    """
    cur = interpolate_thresholds(current_dots, STS_FREQS)
    bas = interpolate_thresholds(baseline_dots, STS_FREQS)
    deltas = []
    for f in STS_FREQS:
        if not np.isnan(cur[f]) and not np.isnan(bas[f]):
            deltas.append(cur[f] - bas[f])
    return float(np.mean(deltas)) if deltas else np.nan


# ─── Features absolues (par record) ──────────────────────────────────────────

def extract_features(row: pd.Series) -> dict:
    """Construit le vecteur de features absolues pour un seul record."""
    left_thresh = interpolate_thresholds(row["dots_left"], STANDARD_FREQS)
    right_thresh = interpolate_thresholds(row["dots_right"], STANDARD_FREQS)

    features = {}
    for freq in STANDARD_FREQS:
        features[f"L_{freq}"] = left_thresh[freq]
        features[f"R_{freq}"] = right_thresh[freq]

    features["PTA_L"] = compute_pta(left_thresh)
    features["PTA_R"] = compute_pta(right_thresh)
    features["high_freq_drop_L"] = compute_high_freq_drop(left_thresh)
    features["high_freq_drop_R"] = compute_high_freq_drop(right_thresh)
    features["asymmetry_mean"] = compute_asymmetry(left_thresh, right_thresh)
    features["hearing_line"] = row.get("hearing_line", np.nan)
    features["evaluation_mode"] = float(row.get("evaluation_mode", 0) or 0)
    features["n_freqs_left"] = float(row.get("n_freqs_left", 0))
    features["n_freqs_right"] = float(row.get("n_freqs_right", 0))

    return features


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Construit la matrice de features absolues pour tout le dataset.

    Retourne (feature_df, feature_cols).
    """
    feature_rows = df.apply(extract_features, axis=1)
    feature_df = pd.DataFrame(list(feature_rows), index=df.index)
    return feature_df, feature_df.columns.tolist()


# ─── Features delta (évolution par patient vs Baseline) ───────────────────────

def build_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les features de changement par rapport au Baseline de chaque patient.

    Pour chaque record Periodic (cat=1) ou Depart (cat=2), calcule le delta
    fréquence par fréquence vs le Baseline (cat=0) du même patient.

    Les records Baseline et les patients sans Baseline retournent NaN.

    Retourne un DataFrame aligné sur l'index de df, avec les colonnes :
      delta_L_{freq}, delta_R_{freq}  — shift en dB par fréquence
      delta_PTA_L, delta_PTA_R       — shift du PTA
      delta_high_freq_drop_L/R       — évolution de la chute HF
      delta_asymmetry                — évolution de l'asymétrie
      sts_L, sts_R                   — Standard Threshold Shift (2000/3000/4000 Hz)
      has_sts_L, has_sts_R           — flag : STS ≥ 10 dB
    """
    nan_row = {f"delta_L_{f}": np.nan for f in STANDARD_FREQS}
    nan_row.update({f"delta_R_{f}": np.nan for f in STANDARD_FREQS})
    nan_row.update({
        "delta_PTA_L": np.nan, "delta_PTA_R": np.nan,
        "delta_high_freq_drop_L": np.nan, "delta_high_freq_drop_R": np.nan,
        "delta_asymmetry": np.nan,
        "sts_L": np.nan, "sts_R": np.nan,
        "has_sts_L": np.nan, "has_sts_R": np.nan,
    })

    # Index : patient → première ligne Baseline (la plus ancienne si plusieurs)
    baselines = (
        df[df["visit_category"] == 0]
        .sort_values("visit_date")
        .drop_duplicates(subset=["patient"], keep="first")
        .set_index("patient")
    )

    rows = []
    for _, row in df.iterrows():
        patient = row["patient"]

        if row["visit_category"] == 0 or patient not in baselines.index:
            rows.append(nan_row.copy())
            continue

        baseline = baselines.loc[patient]

        cur_left = interpolate_thresholds(row["dots_left"], STANDARD_FREQS)
        cur_right = interpolate_thresholds(row["dots_right"], STANDARD_FREQS)
        bas_left = interpolate_thresholds(baseline["dots_left"], STANDARD_FREQS)
        bas_right = interpolate_thresholds(baseline["dots_right"], STANDARD_FREQS)

        delta_left = {
            f: cur_left[f] - bas_left[f]
            for f in STANDARD_FREQS
            if not np.isnan(cur_left[f]) and not np.isnan(bas_left[f])
        }
        delta_right = {
            f: cur_right[f] - bas_right[f]
            for f in STANDARD_FREQS
            if not np.isnan(cur_right[f]) and not np.isnan(bas_right[f])
        }

        sts_l = compute_sts(row["dots_left"], baseline["dots_left"])
        sts_r = compute_sts(row["dots_right"], baseline["dots_right"])

        delta_row = {}
        for f in STANDARD_FREQS:
            delta_row[f"delta_L_{f}"] = delta_left.get(f, np.nan)
            delta_row[f"delta_R_{f}"] = delta_right.get(f, np.nan)

        delta_row["delta_PTA_L"] = compute_pta(delta_left)
        delta_row["delta_PTA_R"] = compute_pta(delta_right)
        delta_row["delta_high_freq_drop_L"] = compute_high_freq_drop(delta_left)
        delta_row["delta_high_freq_drop_R"] = compute_high_freq_drop(delta_right)
        delta_row["delta_asymmetry"] = compute_asymmetry(delta_left, delta_right)
        delta_row["sts_L"] = sts_l
        delta_row["sts_R"] = sts_r
        delta_row["has_sts_L"] = float(sts_l >= 10) if not np.isnan(sts_l) else np.nan
        delta_row["has_sts_R"] = float(sts_r >= 10) if not np.isnan(sts_r) else np.nan

        rows.append(delta_row)

    return pd.DataFrame(rows, index=df.index)


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess(
    feature_df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    imputer: SimpleImputer | None = None,
    fit: bool = True,
) -> tuple[np.ndarray, StandardScaler, SimpleImputer]:
    """
    Impute les NaN puis standardise les features.

    fit=True  : ajuste l'imputer et le scaler (jeu d'entraînement).
    fit=False : applique les transformations existantes (jeu de test).
    """
    if imputer is None:
        imputer = SimpleImputer(strategy="median")
    if scaler is None:
        scaler = StandardScaler()

    if fit:
        X = imputer.fit_transform(feature_df)
        X = scaler.fit_transform(X)
    else:
        X = imputer.transform(feature_df)
        X = scaler.transform(X)

    return X, scaler, imputer
