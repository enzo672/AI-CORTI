"""
Validation par injection synthétique.

Usage :
    python validate_synthetic.py --data CORTI_sample_audiograms_500.json --model-dir results/
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from src.synthetic_validation import (
    build_synthetic_dataset,
    load_trained_models,
    score_synthetic,
    compute_metrics,
    print_synthetic_report,
    save_synthetic_report,
    plot_synthetic_validation,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validation par injection synthétique d'anomalies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data",      required=True,  help="Fichier JSON original (données réelles)")
    parser.add_argument("--model-dir", default="results", help="Dossier contenant les modèles entraînés")
    parser.add_argument("--n-normal",  type=int, default=40, help="Nombre d'audiogrammes normaux synthétiques")
    parser.add_argument("--n-per-type",type=int, default=20, help="Nombre d'anomalies par type")
    parser.add_argument("--no-plots",  action="store_true")
    return parser.parse_args()


def compute_real_ae_threshold(data_path: Path, models: dict) -> float:
    """
    Score les données réelles avec l'AE pour obtenir le seuil de référence
    (percentile 95 des erreurs de reconstruction sur le train set).
    """
    from src.loader import load_json_file
    from src.features import build_feature_matrix, preprocess
    import pandas as pd

    records = load_json_file(data_path)
    df = pd.DataFrame(records).sort_values(["patient", "visit_date"]).reset_index(drop=True)

    feature_df, _ = build_feature_matrix(df)
    X, _, _ = preprocess(feature_df, scaler=models["scaler"],
                          imputer=models["imputer"], fit=False)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        recon = models["ae"](X_tensor).numpy()

    errors = np.mean((X - recon) ** 2, axis=1)
    threshold = float(np.percentile(errors, 95))
    print(f"  Seuil AE (percentile 95 sur données réelles) : {threshold:.4f}")
    return threshold


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    data_path = Path(args.data)

    if not model_dir.exists():
        print(f"Erreur : dossier modèles introuvable → {model_dir}")
        return
    if not data_path.exists():
        print(f"Erreur : fichier données introuvable → {data_path}")
        return

    print("Chargement des modèles...")
    models = load_trained_models(model_dir)

    print("Calcul du seuil AE sur les données réelles...")
    real_ae_threshold = compute_real_ae_threshold(data_path, models)

    print(f"\nGénération des audiogrammes synthétiques "
          f"({args.n_normal} normaux, {args.n_per_type} × 3 types d'anomalies)...")
    synth_df = build_synthetic_dataset(args.n_normal, args.n_per_type)

    print("Scoring avec le modèle entraîné...")
    scores_df = score_synthetic(synth_df, models, real_ae_threshold)

    metrics = compute_metrics(synth_df, scores_df)
    print_synthetic_report(metrics)
    save_synthetic_report(metrics, model_dir)

    if not args.no_plots:
        print("\nGénération des figures...")
        plot_synthetic_validation(metrics, model_dir)

    print("\nTerminé.")


if __name__ == "__main__":
    main()
