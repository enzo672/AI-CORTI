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

# ─── ISO 7029:2017 — Correction normative âge/genre ──────────────────────────
#
# Seuil auditif médian attendu (dB HL) au-dessus de la valeur à 18 ans :
#   H'0.5(freq, age, genre) = a × max(age − 18, 0)^n
#
# Coefficients issus de l'ISO 7029:2017, Annexe A (Table A.1).
# Genre : 1 = homme, 2 = femme (encodage Odyo).
# Plage valide : 18–80 ans (clamp appliqué hors plage).

ISO7029_COEFFS = {
    1: {  # Homme
        250:  {"a": 0.0320, "n": 1.92},
        500:  {"a": 0.0400, "n": 1.92},
        1000: {"a": 0.0640, "n": 1.92},
        2000: {"a": 0.1100, "n": 2.00},
        3000: {"a": 0.2500, "n": 1.90},
        4000: {"a": 0.4800, "n": 1.80},
        6000: {"a": 0.5400, "n": 1.70},
        8000: {"a": 0.2800, "n": 1.90},
    },
    2: {  # Femme
        250:  {"a": 0.0260, "n": 1.88},
        500:  {"a": 0.0400, "n": 2.00},
        1000: {"a": 0.0540, "n": 2.00},
        2000: {"a": 0.0590, "n": 2.12},
        3000: {"a": 0.0980, "n": 2.02},
        4000: {"a": 0.1700, "n": 1.99},
        6000: {"a": 0.3600, "n": 1.82},
        8000: {"a": 0.3400, "n": 1.84},
    },
}

ISO7029_AGE_MIN = 18
ISO7029_AGE_MAX = 80


# ─── ISO 7029 ────────────────────────────────────────────────────────────────

def iso7029_expected(freq: int, age: float, gender: int) -> float:
    """
    Seuil auditif médian attendu (dB HL) selon ISO 7029:2017.

    Représente la perte liée à l'âge et au genre pour un sujet otologiquement
    normal. Utilisé pour calculer des résidus (seuil mesuré − seuil attendu).

    freq   : fréquence en Hz (doit être dans ISO7029_COEFFS)
    age    : âge en années (clampé à [18, 80])
    gender : 1 = homme, 2 = femme
    Retourne NaN si freq ou gender inconnu.
    """
    coeffs = ISO7029_COEFFS.get(gender, {}).get(freq)
    if coeffs is None:
        return np.nan
    theta = max(0.0, min(float(age), ISO7029_AGE_MAX) - ISO7029_AGE_MIN)
    return coeffs["a"] * (theta ** coeffs["n"])


def apply_iso7029_correction(thresholds: dict, age: float, gender: int) -> dict:
    """
    Soustrait le seuil attendu ISO 7029 à chaque fréquence.

    Résidu ≈ 0  → audition normale pour cet âge/genre
    Résidu > 0  → perte au-delà de la norme → suspect
    Résidu < 0  → meilleure audition que la norme

    Les fréquences sans coefficient ISO (ex. 3000 Hz) sont laissées brutes.
    """
    corrected = {}
    for freq, db_val in thresholds.items():
        expected = iso7029_expected(int(freq), age, gender)
        corrected[freq] = (db_val - expected) if not np.isnan(expected) else db_val
    return corrected


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
    """
    Construit le vecteur de features pour un seul record.

    Si age_at_visit et gender sont disponibles, les seuils sont corrigés
    selon ISO 7029:2017 (résidu = mesuré − attendu pour cet âge/genre).
    Sinon, les seuils bruts sont utilisés.
    """
    left_thresh = interpolate_thresholds(row["dots_left"], STANDARD_FREQS)
    right_thresh = interpolate_thresholds(row["dots_right"], STANDARD_FREQS)

    age = row.get("age_at_visit")
    gender = row.get("gender")

    age_valid = age is not None and not (isinstance(age, float) and np.isnan(age))
    gender_valid = (
        gender is not None
        and not (isinstance(gender, float) and np.isnan(gender))
        and int(gender) in ISO7029_COEFFS
    )

    if age_valid and gender_valid:
        left_thresh = apply_iso7029_correction(left_thresh, float(age), int(gender))
        right_thresh = apply_iso7029_correction(right_thresh, float(age), int(gender))

    features = {}
    for freq in STANDARD_FREQS:
        features[f"L_{freq}"] = left_thresh[freq]
        features[f"R_{freq}"] = right_thresh[freq]

    features["PTA_L"] = compute_pta(left_thresh)
    features["PTA_R"] = compute_pta(right_thresh)
    features["high_freq_drop_L"] = compute_high_freq_drop(left_thresh)
    features["high_freq_drop_R"] = compute_high_freq_drop(right_thresh)
    features["asymmetry_mean"] = compute_asymmetry(left_thresh, right_thresh)

    features["age_at_visit"] = float(age) if age_valid else np.nan
    features["gender"] = float(gender) if gender_valid else np.nan
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
