"""
Normalise les 11 rapports JSON du dossier "JSON reports/" sur la structure
exacte de sample.json, puis les écrit dans data/normalized/.

Transformations appliquées :
  - Ordre des clés racine : _id, patient, office, professional, category, version,
    data, visitDate, isDeleted, createdAt, updatedAt, __v, snapshot, submitDate
  - audiogramme : retire showHearingThreshold, déplace sts/msp en fin,
    normalise msp au format {confirmedBy, confirmedByName}
  - divers : garantit la présence de reportDate (null si absent)
  - snapshot.office : retire id/timezone, ajoute fax:null si absent
  - snapshot positionné en fin de document (après data)

Usage :
    python normalize.py
"""

import json
from pathlib import Path

SRC_DIR = Path("JSON reports")
DST_DIR = Path("data/normalized")


def _normalize_msp_side(raw_side: dict) -> dict:
    return {
        "confirmedBy": raw_side.get("confirmedBy", None),
        "confirmedByName": raw_side.get("confirmedByName", ""),
    }


def _normalize_audiogramme(raw: dict) -> dict:
    out = {}
    out["evaluationMode"] = raw.get("evaluationMode", 0)
    out["hearingLine"] = raw.get("hearingLine", "25")
    out["dots"] = raw.get("dots", {"left": [], "right": []})
    out["prevDate"] = raw.get("prevDate", None)
    out["prevReportId"] = raw.get("prevReportId", "")
    out["tables"] = raw.get("tables", {"left": [[]], "right": [[]]})
    out["freeFieldOptions"] = raw.get("freeFieldOptions", {
        "stimulus": [],
        "conditions": [],
        "collaboration": {"value": 0, "explanation": ""},
    })
    out["hulledSounds"] = raw.get("hulledSounds", False)

    sts_raw = raw.get("sts", {}) or {}
    out["sts"] = {
        "right": sts_raw.get("right", None),
        "left": sts_raw.get("left", None),
    }

    msp_raw = raw.get("msp", {}) or {}
    out["msp"] = {
        "right": _normalize_msp_side(msp_raw.get("right") or {}),
        "left": _normalize_msp_side(msp_raw.get("left") or {}),
    }

    return out


def _normalize_divers(raw: dict) -> dict:
    return {
        "testValidity": raw.get("testValidity", 0),
        "audiometer": raw.get("audiometer", {}),
        "currentANSI": raw.get("currentANSI", 0),
        "cc": raw.get("cc", ""),
        "transducers": raw.get("transducers", [1]),
        "reportDate": raw.get("reportDate", None),
        "hearingProtection": raw.get("hearingProtection", None),
    }


def _normalize_office(raw: dict) -> dict:
    return {
        "name": raw.get("name", ""),
        "logo": raw.get("logo", ""),
        "address": raw.get("address", ""),
        "city": raw.get("city", ""),
        "province": raw.get("province", ""),
        "country": raw.get("country", ""),
        "zip": raw.get("zip", ""),
        "email": raw.get("email", ""),
        "phone": raw.get("phone", ""),
        "fax": raw.get("fax", None),
    }


def normalize_record(raw: dict) -> dict:
    raw_data = raw.get("data", {})
    raw_snap = raw.get("snapshot", {})

    return {
        "_id": raw.get("_id", {}),
        "patient": raw.get("patient", ""),
        "office": raw.get("office", {}),
        "professional": raw.get("professional", {}),
        "category": raw.get("category", 0),
        "version": raw.get("version", 1),
        "data": {
            "type": raw_data.get("type", 3),
            "audiogramme": _normalize_audiogramme(raw_data.get("audiogramme", {})),
            "divers": _normalize_divers(raw_data.get("divers", {})),
            "note": raw_data.get("note", []),
        },
        "visitDate": raw.get("visitDate", {}),
        "isDeleted": raw.get("isDeleted", False),
        "createdAt": raw.get("createdAt", {}),
        "updatedAt": raw.get("updatedAt", {}),
        "__v": raw.get("__v", 0),
        "snapshot": {
            "office": _normalize_office(raw_snap.get("office", {})),
            "professional": raw_snap.get("professional", {}),
            "patient": raw_snap.get("patient", ""),
        },
        "submitDate": raw.get("submitDate", {}),
        "syncAt": raw.get("syncAt", None),
    }


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(SRC_DIR.glob("*.json"))
    if not json_files:
        print(f"Aucun fichier JSON trouvé dans {SRC_DIR}/")
        return

    ok, skipped = 0, 0
    for src in json_files:
        try:
            with open(src, encoding="utf-8") as f:
                raw = json.load(f)

            records = raw if isinstance(raw, list) else [raw]
            normalized = [normalize_record(r) for r in records]
            output = normalized if isinstance(raw, list) else normalized[0]

            dst = DST_DIR / src.name
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2, default=str)

            print(f"  OK  {src.name}  ->  {dst}")
            ok += 1
        except Exception as e:
            print(f"  ERR  {src.name}  --  {e}")
            skipped += 1

    print(f"\n{ok} fichiers normalises  |  {skipped} erreurs  ->  {DST_DIR}/")


if __name__ == "__main__":
    main()
