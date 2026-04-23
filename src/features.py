"""
Feature engineering pour les audiogrammes.

Transforme les dicts {fréquence: dB} en vecteurs numériques fixes
adaptés aux modèles ML, avec interpolation aux fréquences standard.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

STANDARD_FREQS = [250, 500, 1000, 2000, 4000, 8000]

PTA_FREQS = [500, 1000, 2000, 4000]


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
        result = {}
        for tf in target_freqs:
            result[tf] = dbs[0] if tf == freqs[0] else np.nan
        return result

    interpolator = interp1d(freqs, dbs, kind="linear", bounds_error=False, fill_value=np.nan)
    return {tf: float(interpolator(tf)) for tf in target_freqs}


def compute_pta(thresholds: dict, pta_freqs: list[int] = PTA_FREQS) -> float:
    """Pure Tone Average : moyenne des seuils aux fréquences de référence."""
    values = [thresholds[f] for f in pta_freqs if not np.isnan(thresholds.get(f, np.nan))]
    return float(np.mean(values)) if values else np.nan


def compute_high_freq_drop(thresholds: dict) -> float:
    """Chute haute fréquence : différence entre 4000 Hz et 8000 Hz."""
    v4 = thresholds.get(4000, np.nan)
    v8 = thresholds.get(8000, np.nan)
    if np.isnan(v4) or np.isnan(v8):
        return np.nan
    return float(v8 - v4)


def compute_asymmetry(left_thresh: dict, right_thresh: dict) -> float:
    """Asymétrie inter-oreilles : moyenne de |OG - OD| sur les fréquences communes."""
    diffs = []
    for f in STANDARD_FREQS:
        l_val = left_thresh.get(f, np.nan)
        r_val = right_thresh.get(f, np.nan)
        if not np.isnan(l_val) and not np.isnan(r_val):
            diffs.append(abs(l_val - r_val))
    return float(np.mean(diffs)) if diffs else np.nan


def extract_features(row: pd.Series) -> dict:
    """
    Construit le vecteur de features pour un seul record.

    Retourne un dict plat avec toutes les features numériques.
    """
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
    Construit la matrice de features pour tout le dataset.

    Retourne :
    - feature_df : DataFrame avec uniquement les colonnes numériques de features
    - feature_cols : liste des noms de colonnes features
    """
    feature_rows = df.apply(extract_features, axis=1)
    feature_df = pd.DataFrame(list(feature_rows))
    feature_cols = feature_df.columns.tolist()
    return feature_df, feature_cols


def preprocess(
    feature_df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    imputer: SimpleImputer | None = None,
    fit: bool = True,
) -> tuple[np.ndarray, StandardScaler, SimpleImputer]:
    """
    Impute les NaN puis standardise les features.

    Si fit=True, ajuste l'imputer et le scaler sur les données fournies.
    Si fit=False, applique des transformations existantes (jeu de test).

    Retourne (X, scaler, imputer).
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
