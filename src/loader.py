"""
Chargement et parsing des fichiers JSON exportés depuis MongoDB (Odyo).

Un record = un rapport d'audiométrie complet.

Format des dots (tableau de 5 valeurs) :
  [0] dB        — seuil auditif mesuré
  [1] fréquence — fréquence en Hz
  [2] catégorie — 1 = points du rapport courant, 3 = points du rapport précédent (gris UI)
  [3] réservé   — toujours 0
  [4] no_response — True = pas de réponse patient (invalide), False = réponse valide

visit_category du record :
  0 = Baseline  (référence initiale du patient)
  1 = Periodic  (suivi régulier)
  2 = Depart    (bilan de sortie)
"""

import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


def _compute_age(dob_str: str | None, visit_date) -> float:
    """Âge en années décimales à la date de visite."""
    if not dob_str or visit_date is None:
        return np.nan
    try:
        dob = pd.to_datetime(dob_str.replace("/", "-"))
        if hasattr(visit_date, "tzinfo") and visit_date.tzinfo is not None:
            dob = dob.tz_localize("UTC")
        return float((visit_date - dob).days / 365.25)
    except Exception:
        return np.nan


def _parse_mongo_value(val):
    """Convertit les types MongoDB extended JSON ($oid, $date) en valeurs Python."""
    if isinstance(val, dict):
        if "$oid" in val:
            return val["$oid"]
        if "$date" in val:
            return pd.to_datetime(val["$date"], utc=True)
    return val


def _parse_dots(dots_list: list) -> dict:
    """
    Transforme la liste de points d'un audiogramme en dict {fréquence_Hz: dB}.

    Format d'un point : [dB, fréquence_Hz, dot_category, réservé, no_response]
    - dot_category == 1 : points du rapport courant (seuls conservés)
    - dot_category == 3 : points du rapport précédent (affichés en gris, ignorés)
    - no_response == True : pas de réponse du patient → point exclu
    """
    result = {}
    for point in dots_list:
        if len(point) < 2:
            continue
        db_value, freq = point[0], point[1]
        dot_category = point[2] if len(point) > 2 else 1
        no_response = point[4] if len(point) > 4 else False

        if dot_category != 1:
            continue
        if no_response or db_value is None:
            continue

        result[float(freq)] = float(db_value)
    return result


_DB_MIN = -10.0
_DB_MAX = 120.0
_MIN_FREQS_PER_EAR = 4


def _is_test_audiogram(dots_left: dict, dots_right: dict) -> bool:
    """
    Détecte les audiogrammes de test/calibration non cliniques.

    Trois critères (un seul suffit à rejeter) :
    1. Valeur hors range physiologique (< -10 ou > 120 dB HL)
    2. Moins de 4 fréquences mesurées sur l'une ou l'autre oreille
    3. Audiogramme anormalement plat près de zéro :
       std < 2 dB ET tous les seuils ≤ 10 dB → sweep de calibration
    """
    all_vals = list(dots_left.values()) + list(dots_right.values())
    if not all_vals:
        return True

    if any(v < _DB_MIN or v > _DB_MAX for v in all_vals):
        return True

    if len(dots_left) < _MIN_FREQS_PER_EAR or len(dots_right) < _MIN_FREQS_PER_EAR:
        return True

    if max(all_vals) <= 10.0 and float(np.std(all_vals)) < 2.0:
        return True

    return False


def load_record(record: dict) -> dict:
    """
    Parse un record JSON en dict plat exploitable pour le ML.

    Supporte deux formats :
    - Ancien format MongoDB (présence de "_id", "isDeleted", "testValidity")
    - Nouveau format CORTI (category + visitDate ISO + snapshot.patient)

    Retourne None si le record est supprimé ou invalide (ancien format uniquement).
    """
    is_mongodb = "_id" in record

    if is_mongodb:
        if record.get("isDeleted", False):
            return None
        data = record.get("data", {})
        divers = data.get("divers", {})
        if divers.get("testValidity", 0) != 0:
            return None
        audiogramme = data.get("audiogramme", {})

        visit_date = _parse_mongo_value(record.get("visitDate", {}))
        patient = record.get("patient")
        record_id = _parse_mongo_value(record.get("_id", {}))
        gender = None
        age_at_visit = np.nan
    else:
        # Format CORTI : visitDate ISO, démographie dans snapshot.patient
        data = record.get("data") or {}
        audiogramme = data.get("audiogramme", {})

        visit_date_raw = record.get("visitDate")
        visit_date = pd.to_datetime(visit_date_raw, utc=True) if visit_date_raw else None

        snapshot = record.get("snapshot") or {}
        patient_info = snapshot.get("patient") or {}
        dob = patient_info.get("dob") or ""
        gender_raw = patient_info.get("gender")
        # Encodage Odyo : 1 = homme, 2 = femme
        gender = int(gender_raw) if gender_raw is not None else None
        age_at_visit = _compute_age(dob, visit_date)

        # Identifiant patient synthétique (dob + genre — meilleure approximation sans _id)
        patient = f"{dob}_{gender}"
        record_id = None

    dots = audiogramme.get("dots", {})
    dots_left = _parse_dots(dots.get("left", []))
    dots_right = _parse_dots(dots.get("right", []))

    if _is_test_audiogram(dots_left, dots_right):
        return None

    return {
        "record_id": record_id,
        "patient": patient,
        "visit_category": record.get("category"),   # 0=Baseline, 1=Periodic, 2=Depart
        "visit_date": visit_date,
        "gender": gender,          # 1=homme, 2=femme (encodage Odyo)
        "age_at_visit": age_at_visit,
        "dots_left": dots_left,
        "dots_right": dots_right,
        "n_freqs_left": len(dots_left),
        "n_freqs_right": len(dots_right),
    }


def load_json_file(path: Union[str, Path]) -> list[dict]:
    """
    Charge un fichier JSON (un record ou une liste de records) et retourne
    une liste de records parsés, chacun annoté avec son fichier source.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = raw if isinstance(raw, list) else [raw]
    parsed = [load_record(r) for r in records]
    result = [r for r in parsed if r is not None]
    for r in result:
        r["source_file"] = path.name
    return result


def load_dataset(data_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Charge tous les fichiers JSON d'un dossier et retourne un DataFrame.

    Chaque ligne = un rapport d'audiométrie valide.
    Les records sont triés par patient puis par visit_date pour faciliter
    l'analyse temporelle (Baseline → Periodic → Depart).
    La colonne source_file indique le fichier d'origine de chaque record.
    """
    data_dir = Path(data_dir)
    all_records = []

    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        raise ValueError(f"Aucun fichier JSON trouvé dans {data_dir}")

    for json_file in json_files:
        records = load_json_file(json_file)
        print(f"  {json_file.name} : {len(records)} records")
        all_records.extend(records)

    if not all_records:
        raise ValueError(f"Aucun record valide trouvé dans {data_dir}")

    df = pd.DataFrame(all_records)

    # Déduplication : par record_id (MongoDB) ou par clé fonctionnelle (CORTI)
    if df["record_id"].notna().any():
        df = df.drop_duplicates(subset=["record_id"])
    else:
        df = df.drop_duplicates(subset=["patient", "visit_date", "visit_category"])

    df = df.sort_values(["patient", "visit_date"]).reset_index(drop=True)
    return df
