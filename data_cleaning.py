"""
Script de nettoyage des données avant entraînement.

Usage :
    python data_cleaning.py --data data/
    python data_cleaning.py --data data/ --export-rejected
    python data_cleaning.py --data CORTI_sample_audiograms_500.json
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nettoyage et audit des audiogrammes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="Fichier JSON ou dossier data/")
    parser.add_argument(
        "--export-rejected",
        action="store_true",
        help="Exporte les records rejetés dans data/rejected_records.csv",
    )
    return parser.parse_args()


def load_raw(data_path: Path) -> pd.DataFrame:
    from src.loader import load_dataset, load_json_file

    # Désactiver le filtre test du loader pour laisser cleaning.py tout voir
    import src.loader as _loader
    original_filter = _loader._is_test_audiogram
    _loader._is_test_audiogram = lambda L, R: False  # bypass temporaire

    if data_path.is_dir():
        df = load_dataset(data_path)
    else:
        records = load_json_file(data_path)
        df = pd.DataFrame(records).sort_values(["patient", "visit_date"]).reset_index(drop=True)

    _loader._is_test_audiogram = original_filter  # restaurer
    return df


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)

    if not data_path.exists():
        print(f"Erreur : chemin introuvable → {data_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Chargement : {data_path}")
    df = load_raw(data_path)
    print(f"  {len(df)} records chargés")

    from src.cleaning import clean_dataset, cleaning_report
    clean_df, rejected_df = clean_dataset(df)
    cleaning_report(df, clean_df, rejected_df)

    # Détail des rejetés
    if not rejected_df.empty:
        print("Détail des records rejetés :")
        for _, row in rejected_df.iterrows():
            src = row.get("source_file", "")
            pat = str(row.get("patient", ""))[:20]
            date = str(row.get("visit_date", ""))[:10]
            print(f"  [{src}] patient={pat} date={date} → {row['rejection_reasons']}")

    # Détail des soft flags
    soft = clean_df[clean_df.get("aberrant_flags", pd.Series("", index=clean_df.index)) != ""]
    if not soft.empty:
        print(f"\nRecords avec points aberrants conservés ({len(soft)}) :")
        for _, row in soft.head(20).iterrows():
            src = row.get("source_file", "")
            pat = str(row.get("patient", ""))[:20]
            print(f"  [{src}] patient={pat} → {row['aberrant_flags']}")
        if len(soft) > 20:
            print(f"  ... et {len(soft) - 20} autres")

    if args.export_rejected and not rejected_df.empty:
        out_path = Path("data/rejected_records.csv")
        export_cols = ["source_file", "patient", "visit_date", "visit_category", "rejection_reasons"]
        export_cols = [c for c in export_cols if c in rejected_df.columns]
        rejected_df[export_cols].to_csv(out_path, index=False)
        print(f"\nRejetés exportés → {out_path}")


if __name__ == "__main__":
    main()
