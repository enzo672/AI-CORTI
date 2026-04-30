"""
Nettoyage et validation des audiogrammes avant entraînement.

Deux niveaux de contrôle :
- Record-level : le record entier est rejeté (trop peu de points, hors range, etc.)
- Point-level  : un point individuel est aberrant (déviation par rapport aux voisins)

Usage :
    from src.cleaning import clean_dataset
    clean_df, rejected_df = clean_dataset(df)
"""

import numpy as np
import pandas as pd

# ─── Paramètres ──────────────────────────────────────────────────────────────

DB_MIN: float = -10.0
DB_MAX: float = 120.0
MIN_FREQS_PER_EAR: int = 4
CALIBRATION_MAX_DB: float = 10.0
CALIBRATION_MAX_STD: float = 2.0
ABERRANT_NEIGHBOR_DIFF_DB: float = 35.0  # écart max vs interpolation des voisins


# ─── Détection point aberrant ─────────────────────────────────────────────────

def _find_aberrant_points(thresholds: dict, diff_threshold: float = ABERRANT_NEIGHBOR_DIFF_DB) -> list[float]:
    """
    Retourne les fréquences dont le seuil s'écarte trop de l'interpolation
    linéaire (échelle log-fréquence) de ses deux voisins mesurés.

    Un seuil aberrant ressemble à un artefact de mesure, pas à une pathologie.
    """
    if len(thresholds) < 3:
        return []

    freqs = sorted(thresholds.keys())
    log_freqs = np.log2(freqs)
    values = [thresholds[f] for f in freqs]
    aberrant = []

    for i in range(1, len(freqs) - 1):
        # Interpolation linéaire entre le voisin gauche et droit
        t = (log_freqs[i] - log_freqs[i - 1]) / (log_freqs[i + 1] - log_freqs[i - 1])
        expected = values[i - 1] + t * (values[i + 1] - values[i - 1])
        if abs(values[i] - expected) > diff_threshold:
            aberrant.append(freqs[i])

    return aberrant


# ─── Validation record ────────────────────────────────────────────────────────

def audit_record(row: pd.Series) -> list[str]:
    """
    Vérifie un record et retourne la liste des problèmes détectés.
    Liste vide = record valide.
    """
    issues = []
    dots_left: dict  = row.get("dots_left",  {}) or {}
    dots_right: dict = row.get("dots_right", {}) or {}
    all_vals = list(dots_left.values()) + list(dots_right.values())

    # 1. Données vides
    if not all_vals:
        issues.append("no_data")
        return issues

    # 2. Couverture fréquentielle insuffisante
    if len(dots_left) < MIN_FREQS_PER_EAR:
        issues.append(f"too_few_freqs_left:{len(dots_left)}")
    if len(dots_right) < MIN_FREQS_PER_EAR:
        issues.append(f"too_few_freqs_right:{len(dots_right)}")

    # 3. Valeurs hors range physiologique
    out_of_range = [v for v in all_vals if v < DB_MIN or v > DB_MAX]
    if out_of_range:
        issues.append(f"out_of_range:{out_of_range[:3]}")

    # 4. Audiogramme de calibration (parfaitement plat près de zéro)
    if all_vals and max(all_vals) <= CALIBRATION_MAX_DB and np.std(all_vals) < CALIBRATION_MAX_STD:
        issues.append("calibration_sweep")

    # 5. Points aberrants (artefacts de mesure)
    aberrant_L = _find_aberrant_points(dots_left)
    aberrant_R = _find_aberrant_points(dots_right)
    if aberrant_L:
        issues.append(f"aberrant_point_left:{aberrant_L}")
    if aberrant_R:
        issues.append(f"aberrant_point_right:{aberrant_R}")

    # 6. Démographie manquante
    if pd.isna(row.get("visit_date")):
        issues.append("missing_visit_date")
    if not row.get("patient") or str(row.get("patient")).startswith("_"):
        issues.append("missing_patient_id")

    return issues


# ─── Pipeline de nettoyage ────────────────────────────────────────────────────

# Issues qui entraînent un rejet complet du record
_HARD_REJECT = {"no_data", "calibration_sweep", "missing_visit_date", "too_few_freqs_left", "too_few_freqs_right", "out_of_range"}


def _is_hard_reject(issues: list[str]) -> bool:
    for issue in issues:
        for hard in _HARD_REJECT:
            if issue.startswith(hard):
                return True
    return False


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sépare les records valides des records problématiques.

    Retourne :
    - clean_df    : records prêts pour l'entraînement
    - rejected_df : records écartés avec colonne 'rejection_reasons'

    Les records avec uniquement des points aberrants (soft flag) sont conservés
    dans clean_df mais annotés dans 'aberrant_flags' pour traçabilité.
    """
    all_issues = df.apply(audit_record, axis=1)

    hard_mask   = all_issues.apply(_is_hard_reject)
    soft_mask   = (~hard_mask) & all_issues.apply(lambda x: bool(x))

    clean_df    = df[~hard_mask].copy()
    rejected_df = df[hard_mask].copy()

    rejected_df["rejection_reasons"] = all_issues[hard_mask].apply(lambda x: " | ".join(x))
    clean_df["aberrant_flags"]       = all_issues[soft_mask].apply(lambda x: " | ".join(x))
    clean_df["aberrant_flags"]       = clean_df.get("aberrant_flags", pd.Series("", index=clean_df.index)).fillna("")

    return clean_df.reset_index(drop=True), rejected_df.reset_index(drop=True)


def cleaning_report(df: pd.DataFrame, clean_df: pd.DataFrame, rejected_df: pd.DataFrame) -> None:
    """Affiche un rapport synthétique du nettoyage."""
    n_total    = len(df)
    n_clean    = len(clean_df)
    n_rejected = len(rejected_df)
    n_soft     = (clean_df.get("aberrant_flags", pd.Series("", index=clean_df.index)) != "").sum()

    print(f"\n{'='*50}")
    print(f"Rapport de nettoyage")
    print(f"{'='*50}")
    print(f"  Total records chargés : {n_total}")
    print(f"  Rejetés (hard)        : {n_rejected} ({100*n_rejected/n_total:.1f}%)")
    print(f"  Flaggés soft          : {n_soft}     ({100*n_soft/n_total:.1f}%)")
    print(f"  Conservés propres     : {n_clean}   ({100*n_clean/n_total:.1f}%)")

    if n_rejected > 0:
        print(f"\n  Raisons de rejet :")
        all_reasons = rejected_df["rejection_reasons"].str.split(" | ").explode()
        reason_prefix = all_reasons.str.split(":").str[0]
        for reason, count in reason_prefix.value_counts().items():
            print(f"    {reason:30s}: {count}")

    if n_soft > 0:
        print(f"\n  Flags soft (points aberrants conservés) :")
        soft_reasons = clean_df["aberrant_flags"][clean_df["aberrant_flags"] != ""].str.split(" | ").explode()
        soft_prefix = soft_reasons.str.split(":").str[0]
        for reason, count in soft_prefix.value_counts().items():
            print(f"    {reason:30s}: {count}")
    print(f"{'='*50}\n")
