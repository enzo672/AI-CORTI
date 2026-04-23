"""
Chargement et parsing des fichiers JSON exportés depuis MongoDB (Odyo).

Un record = un rapport d'audiométrie complet.
"""

import json
import os
from pathlib import Path
from typing import Union

import pandas as pd


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

    Format d'un point : [dB, fréquence_Hz, type_transducteur, ?, masked]
    On garde uniquement les points non masqués (masked == False).
    """
    result = {}
    for point in dots_list:
        if len(point) < 2:
            continue
        db_value, freq = point[0], point[1]
        masked = point[4] if len(point) > 4 else False
        if not masked and db_value is not None:
            result[float(freq)] = float(db_value)
    return result


def load_record(record: dict) -> dict:
    """
    Parse un record JSON MongoDB en dict plat exploitable pour le ML.

    Retourne None si le record est supprimé ou invalide.
    """
    if record.get("isDeleted", False):
        return None

    data = record.get("data", {})
    audiogramme = data.get("audiogramme", {})
    divers = data.get("divers", {})

    test_validity = divers.get("testValidity", 0)
    if test_validity != 0:
        return None

    dots = audiogramme.get("dots", {})
    dots_left = _parse_dots(dots.get("left", []))
    dots_right = _parse_dots(dots.get("right", []))

    hearing_line = audiogramme.get("hearingLine")
    try:
        hearing_line = float(hearing_line) if hearing_line not in (None, "") else None
    except (ValueError, TypeError):
        hearing_line = None

    transducers = divers.get("transducers", [])

    return {
        "record_id": _parse_mongo_value(record.get("_id", {})),
        "patient": record.get("patient"),
        "category": record.get("category"),
        "data_type": data.get("type"),
        "version": record.get("version"),
        "visit_date": _parse_mongo_value(record.get("visitDate", {})),
        "report_date": divers.get("reportDate"),
        "evaluation_mode": audiogramme.get("evaluationMode"),
        "hearing_line": hearing_line,
        "transducers": transducers,
        "dots_left": dots_left,
        "dots_right": dots_right,
        "n_freqs_left": len(dots_left),
        "n_freqs_right": len(dots_right),
    }


def load_json_file(path: Union[str, Path]) -> list[dict]:
    """
    Charge un fichier JSON (un record ou une liste de records) et retourne
    une liste de records parsés.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = raw if isinstance(raw, list) else [raw]
    parsed = [load_record(r) for r in records]
    return [r for r in parsed if r is not None]


def load_dataset(data_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Charge tous les fichiers JSON d'un dossier et retourne un DataFrame.

    Chaque ligne = un rapport d'audiométrie valide.
    """
    data_dir = Path(data_dir)
    all_records = []
    for json_file in sorted(data_dir.glob("*.json")):
        all_records.extend(load_json_file(json_file))

    if not all_records:
        raise ValueError(f"Aucun record valide trouvé dans {data_dir}")

    df = pd.DataFrame(all_records)
    df = df.drop_duplicates(subset=["record_id"])
    df = df.reset_index(drop=True)
    return df
