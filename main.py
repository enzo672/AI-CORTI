"""
Point d'entrée CLI pour le pipeline de détection d'anomalies sur audiogrammes.

Usage :
    python main.py --data sample.json
    python main.py --data data/raw/ --mode delta --epochs 200 --output-dir results/
    python main.py --data sample.json --no-plots --contamination 0.1
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Détection d'anomalies non supervisée sur audiogrammes Odyo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Fichier JSON ou dossier contenant les exports MongoDB",
    )
    parser.add_argument(
        "--mode",
        choices=["abs", "delta"],
        default="abs",
        help=(
            "abs  : features absolues (tous les records)\n"
            "delta: features d'évolution vs Baseline (Periodic/Depart uniquement)"
        ),
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Proportion estimée d'anomalies pour l'Isolation Forest",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Nombre d'epochs pour l'Autoencoder",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Dossier de sortie (modèles, scores CSV, figures)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Désactive la génération des figures",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device PyTorch : 'cpu' ou 'cuda'",
    )
    return parser.parse_args()


def load_data(data_path: Path) -> pd.DataFrame:
    from src.loader import load_dataset, load_json_file

    if data_path.is_dir():
        print(f"Chargement du dossier : {data_path}")
        df = load_dataset(data_path)
    else:
        print(f"Chargement du fichier : {data_path}")
        records = load_json_file(data_path)
        df = pd.DataFrame(records)
        df = df.sort_values(["patient", "visit_date"]).reset_index(drop=True)

    print(f"  {len(df)} records valides — {df['patient'].nunique()} patients uniques")
    return df


def build_features(df: pd.DataFrame, mode: str):
    from src.features import build_delta_features, build_feature_matrix, preprocess

    print(f"\nConstruction des features ({mode})...")

    if mode == "abs":
        feature_df, feature_cols = build_feature_matrix(df)
        X, scaler, imputer = preprocess(feature_df, fit=True)
        df_pipeline = df
        print(f"  Shape matrice : {X.shape} — {len(feature_cols)} features")
        return X, df_pipeline, scaler, imputer

    # mode == "delta"
    feature_df, feature_cols = build_feature_matrix(df)
    delta_df = build_delta_features(df)
    delta_mask = delta_df["delta_PTA_L"].notna()
    n_delta = delta_mask.sum()
    print(f"  Records avec delta calculable : {n_delta} / {len(df)}")

    if n_delta < 5:
        print(
            "  Pas assez de records delta (< 5). "
            "Vérifiez que le dataset contient des Baseline + Periodic/Depart.\n"
            "  Bascule automatique sur le mode 'abs'.",
            file=sys.stderr,
        )
        X, scaler, imputer = preprocess(feature_df, fit=True)
        df_pipeline = df
    else:
        X, scaler, imputer = preprocess(delta_df[delta_mask], fit=True)
        df_pipeline = df[delta_mask].reset_index(drop=True)

    print(f"  Shape matrice : {X.shape}")
    return X, df_pipeline, scaler, imputer


def save_outputs(
    output_dir: Path,
    scores_df: pd.DataFrame,
    if_model,
    ae_model,
    scaler,
    imputer,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_path = output_dir / "scores.csv"
    scores_df.to_csv(scores_path, index=True)
    print(f"\nScores sauvegardés → {scores_path}")

    joblib.dump(if_model, output_dir / "isolation_forest.joblib")
    joblib.dump(scaler,   output_dir / "scaler.joblib")
    joblib.dump(imputer,  output_dir / "imputer.joblib")
    torch.save(ae_model.state_dict(), output_dir / "autoencoder.pt")
    print(f"Modèles sauvegardés → {output_dir}/")


def generate_plots(
    output_dir: Path,
    df_pipeline: pd.DataFrame,
    scores_df: pd.DataFrame,
    X: np.ndarray,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.evaluate import (
        plot_anomaly_score_distribution,
        plot_top_anomalies,
        plot_umap,
    )

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Distribution des scores
    _, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_anomaly_score_distribution(
        scores_df["anomaly_score_if"], title="Isolation Forest", ax=axes[0]
    )
    plot_anomaly_score_distribution(
        scores_df["reconstruction_error"], title="Autoencoder", ax=axes[1]
    )
    plot_anomaly_score_distribution(
        scores_df["pca_reconstruction_error"], title="PCA baseline", ax=axes[2]
    )
    plt.tight_layout()
    plt.savefig(figures_dir / "score_distributions.png", dpi=120)
    plt.close()

    # Top anomalies
    plot_top_anomalies(
        df_pipeline,
        scores_df["reconstruction_error"],
        n=min(6, len(scores_df)),
        score_col="reconstruction_error",
    )
    plt.savefig(figures_dir / "top_anomalies.png", dpi=120)
    plt.close()

    # UMAP (peut être lent sur grands datasets)
    try:
        plot_umap(
            X,
            scores_df["reconstruction_error"].values,
            title="UMAP — Erreur de reconstruction",
            label_name="Reconstruction error",
        )
        plt.savefig(figures_dir / "umap_reconstruction.png", dpi=120)
        plt.close()

        plot_umap(
            X,
            df_pipeline["visit_category"].values,
            title="UMAP — Catégorie de visite",
            label_name="Visite",
            is_categorical=True,
        )
        plt.savefig(figures_dir / "umap_visit_category.png", dpi=120)
        plt.close()
    except Exception as e:
        print(f"  UMAP ignoré : {e}")

    print(f"Figures sauvegardées → {figures_dir}/")


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)

    if not data_path.exists():
        print(f"Erreur : chemin introuvable → {data_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)

    df = load_data(data_path)
    X, df_pipeline, scaler, imputer = build_features(df, args.mode)

    print("\nLancement du pipeline non supervisé...")
    from src.models.unsupervised import run_unsupervised_pipeline

    scores_df, if_model, ae_model = run_unsupervised_pipeline(
        X,
        contamination=args.contamination,
        ae_epochs=args.epochs,
        device=args.device,
    )

    print("\n=== Résultats ===")
    from src.evaluate import summary_report
    summary_report(df_pipeline, scores_df)

    save_outputs(output_dir, scores_df, if_model, ae_model, scaler, imputer)

    if not args.no_plots:
        print("\nGénération des figures...")
        generate_plots(output_dir, df_pipeline, scores_df, X)

    print("\nTerminé.")


if __name__ == "__main__":
    main()
