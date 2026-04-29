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

STANDARD_FREQS = [250, 500, 1000, 2000, 3000, 4000, 6000, 8000]

PTA_FREQS = [500, 1000, 2000, 4000]

# Barany Society 2015 : PTA basses fréquences ≥ 25 dB HL → patron Ménière
LOW_FREQ_PTA_FREQS = [250, 500, 1000]

# Standard Threshold Shift (OSHA) : moyenne à 2000, 3000, 4000 Hz
# ≥ 10 dB = shift cliniquement significatif
STS_FREQS = [2000, 3000, 4000]

VISIT_CATEGORY_LABELS = {0: "Baseline", 1: "Periodic", 2: "Depart"}

# ─── Correction normative âge/genre (OSHA 29 CFR 1910.95 Appendix F) ─────────
#
# Tables de correction en dB HL pour chaque fréquence, âge et genre.
# Source : OSHA Appendix F (Table F-1 hommes, Table F-2 femmes).
# Couverture : 1000–6000 Hz, âges 20–60 ans.
#
# Pour 250, 500, 8000 Hz (hors table OSHA) : estimations cliniques
# basées sur la littérature audiométrique (effet d'âge moindre en BF,
# légèrement inférieur à 6000 Hz en HF).
#
# Genre : 1 = homme, 2 = femme (encodage Odyo).
# Usage : residual = seuil_mesuré − correction_attendue(age, freq, gender)
#   ≈ 0   → audition normale pour cet âge/genre
#   > 0   → perte au-delà de la norme → suspect
#   < 0   → meilleure audition que la norme

OSHA_CORRECTIONS = {
    1: {  # Hommes — Table F-1
        #  freq :  {age: correction_dB}
        250:  {20: 2,  25: 2,  30: 3,  35: 3,  40: 4,  45: 5,  50: 6,  55: 7,  60: 8},
        500:  {20: 3,  25: 3,  30: 4,  35: 4,  40: 5,  45: 6,  50: 7,  55: 8,  60: 9},
        1000: {20: 5,  25: 5,  30: 6,  35: 7,  40: 7,  45: 8,  50: 9,  55: 10, 60: 11},
        2000: {20: 3,  25: 3,  30: 4,  35: 5,  40: 6,  45: 7,  50: 9,  55: 11, 60: 13},
        3000: {20: 4,  25: 5,  30: 6,  35: 8,  40: 10, 45: 13, 50: 16, 55: 19, 60: 23},
        4000: {20: 5,  25: 7,  30: 9,  35: 11, 40: 14, 45: 18, 50: 22, 55: 27, 60: 33},
        6000: {20: 8,  25: 10, 30: 12, 35: 15, 40: 19, 45: 23, 50: 27, 55: 32, 60: 38},
        8000: {20: 7,  25: 9,  30: 11, 35: 14, 40: 17, 45: 21, 50: 25, 55: 29, 60: 34},
    },
    2: {  # Femmes — Table F-2
        250:  {20: 2,  25: 2,  30: 3,  35: 3,  40: 3,  45: 4,  50: 4,  55: 5,  60: 6},
        500:  {20: 3,  25: 3,  30: 4,  35: 4,  40: 4,  45: 5,  50: 5,  55: 6,  60: 7},
        1000: {20: 7,  25: 8,  30: 8,  35: 9,  40: 10, 45: 11, 50: 12, 55: 13, 60: 14},
        2000: {20: 4,  25: 5,  30: 6,  35: 6,  40: 7,  45: 8,  50: 10, 55: 11, 60: 12},
        3000: {20: 3,  25: 4,  30: 5,  35: 7,  40: 8,  45: 10, 50: 11, 55: 14, 60: 16},
        4000: {20: 3,  25: 4,  30: 5,  35: 7,  40: 8,  45: 10, 50: 12, 55: 14, 60: 17},
        6000: {20: 6,  25: 7,  30: 9,  35: 11, 40: 13, 45: 15, 50: 17, 55: 19, 60: 22},
        8000: {20: 6,  25: 7,  30: 8,  35: 10, 40: 12, 45: 14, 50: 16, 55: 18, 60: 21},
    },
}

OSHA_AGE_MIN = 20
OSHA_AGE_MAX = 60


def age_correction_expected(freq: int, age: float, gender: int) -> float:
    """
    Seuil auditif attendu (dB HL) pour un sujet normal de cet âge et genre.

    Source : OSHA 29 CFR 1910.95 Appendix F (Table F-1 hommes, F-2 femmes)
    pour 1000–6000 Hz, estimations cliniques pour 250, 500, 8000 Hz.
    Interpolation linéaire entre les points de la table.
    Age clampé à [20, 60].

    Retourne NaN si freq ou gender inconnu.
    """
    table = OSHA_CORRECTIONS.get(gender, {}).get(freq)
    if table is None:
        return np.nan

    age_clamped = max(OSHA_AGE_MIN, min(float(age), OSHA_AGE_MAX))
    ages = sorted(table.keys())

    if age_clamped <= ages[0]:
        return float(table[ages[0]])
    if age_clamped >= ages[-1]:
        return float(table[ages[-1]])

    for i in range(len(ages) - 1):
        a0, a1 = ages[i], ages[i + 1]
        if a0 <= age_clamped <= a1:
            t = (age_clamped - a0) / (a1 - a0)
            return float(table[a0] + t * (table[a1] - table[a0]))

    return np.nan


def apply_age_correction(thresholds: dict, age: float, gender: int) -> dict:
    """
    Soustrait le seuil attendu (OSHA Appendix F) à chaque fréquence.

    Résidu ≈ 0  → audition normale pour cet âge/genre
    Résidu > 0  → perte au-delà de la norme → suspect
    Résidu < 0  → meilleure audition que la norme

    Les fréquences sans entrée dans la table sont laissées brutes.
    """
    corrected = {}
    for freq, db_val in thresholds.items():
        expected = age_correction_expected(int(freq), age, gender)
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


def compute_notch_derivative(
    thresholds: dict, freq_lo: int = 2000, freq_hi: int = 8000
) -> tuple[float, float]:
    """
    Détecte l'encoche la plus profonde entre freq_lo et freq_hi par dérivée discrète.

    Principe :
    - Dérivée 1ère normalisée par octave : dT / d(log2 f)
    - Un changement de signe + → − signale un maximum local (creux audiométrique NIHL)
    - Profondeur = valeur_pic − moyenne(voisin_gauche, voisin_droit)

    Avantage vs formule fixe à 4kHz : capte les encoches à 3, 4 ou 6 kHz sans hypothèse
    sur la fréquence, et reste robuste si la récupération à 8kHz est absente.

    Retourne (notch_depth, notch_freq) :
    - notch_depth : profondeur en dB (0.0 si aucun creux détecté)
    - notch_freq  : fréquence Hz du creux, np.nan si aucun
    """
    freqs = sorted([
        f for f in thresholds
        if freq_lo <= f <= freq_hi and not np.isnan(thresholds.get(f, np.nan))
    ])

    if len(freqs) < 3:
        return 0.0, np.nan

    vals  = np.array([thresholds[f] for f in freqs], dtype=float)
    log_f = np.log2(np.array(freqs, dtype=float))
    d1    = np.diff(vals) / np.diff(log_f)

    best_depth = 0.0
    best_freq  = np.nan

    for i in range(len(d1) - 1):
        if d1[i] > 0 and d1[i + 1] < 0:
            depth = vals[i + 1] - (vals[i] + vals[i + 2]) / 2
            if depth > best_depth:
                best_depth = depth
                best_freq  = float(freqs[i + 1])

    return best_depth, best_freq


def compute_low_freq_pta(thresholds: dict) -> float:
    """
    PTA basses fréquences (250, 500, 1000 Hz) en seuils absolus (dB HL).

    Doit être calculé sur les seuils bruts AVANT correction ISO 7029.
    Critère Barany Society 2015 : valeur ≥ 25 dB HL sur l'oreille atteinte
    = patron audiométrique de la maladie de Ménière.
    """
    values = [thresholds[f] for f in LOW_FREQ_PTA_FREQS
              if not np.isnan(thresholds.get(f, np.nan))]
    return float(np.mean(values)) if values else np.nan


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
    left_thresh_raw  = interpolate_thresholds(row["dots_left"],  STANDARD_FREQS)
    right_thresh_raw = interpolate_thresholds(row["dots_right"], STANDARD_FREQS)

    age = row.get("age_at_visit")
    gender = row.get("gender")

    age_valid = age is not None and not (isinstance(age, float) and np.isnan(age))
    gender_valid = (
        gender is not None
        and not (isinstance(gender, float) and np.isnan(gender))
        and int(gender) in OSHA_CORRECTIONS
    )

    if age_valid and gender_valid:
        left_thresh  = apply_age_correction(left_thresh_raw,  float(age), int(gender))
        right_thresh = apply_age_correction(right_thresh_raw, float(age), int(gender))
    else:
        left_thresh  = left_thresh_raw
        right_thresh = right_thresh_raw

    features = {}
    for freq in STANDARD_FREQS:
        features[f"L_{freq}"] = left_thresh[freq]
        features[f"R_{freq}"] = right_thresh[freq]

    features["PTA_L"] = compute_pta(left_thresh)
    features["PTA_R"] = compute_pta(right_thresh)
    features["high_freq_drop_L"] = compute_high_freq_drop(left_thresh)
    features["high_freq_drop_R"] = compute_high_freq_drop(right_thresh)
    features["notch_depth_L"], features["notch_freq_L"] = compute_notch_derivative(left_thresh)
    features["notch_depth_R"], features["notch_freq_R"] = compute_notch_derivative(right_thresh)
    # Seuils absolus avant correction — critère Barany Society 2015 (Ménière)
    features["low_freq_pta_L"] = compute_low_freq_pta(left_thresh_raw)
    features["low_freq_pta_R"] = compute_low_freq_pta(right_thresh_raw)
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
