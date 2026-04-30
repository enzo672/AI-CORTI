"""
Point d'entrée CLI pour le pipeline de détection d'anomalies sur audiogrammes.

Usage :
    python main.py --data data/clean_dataset.pkl   ← recommandé après 04_data_cleaning.ipynb
    python main.py --data data/                    ← charge + nettoie à la volée
    python main.py --data data/ --mode delta --epochs 200 --output-dir results/

Le dossier data/ peut contenir plusieurs fichiers ..._01.json, ..._02.json, etc.
Ils sont fusionnés automatiquement avec déduplication et tracking par source_file.
Si un fichier .pkl est fourni (produit par 04_data_cleaning.ipynb), le nettoyage
est ignoré — les données sont déjà propres.
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
        help="clean_dataset.json (produit par 04_data_cleaning.ipynb), fichier .json, ou dossier data/",
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
        default=None,
        help="Dossier de sortie (modèles, scores CSV, figures). Défaut : results/<mode>",
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


def load_data(data_path: Path) -> tuple[pd.DataFrame, bool]:
    """
    Retourne (df, already_clean).
    already_clean=True si les données viennent d'un .pkl (déjà nettoyées).
    """
    if data_path.name == "clean_dataset.json":
        print(f"Chargement dataset propre : {data_path}")
        df = pd.read_json(data_path, orient="records")
        # Les clés de dots sont sérialisées en string par JSON → reconvertir en float
        for col in ("dots_left", "dots_right"):
            df[col] = df[col].apply(
                lambda d: {float(k): v for k, v in d.items()} if isinstance(d, dict) else {}
            )
        df["visit_date"] = pd.to_datetime(df["visit_date"], utc=True)
        print(f"  {len(df)} records — {df['patient'].nunique()} patients (nettoyage ignoré)")
        return df, True

    from src.loader import load_dataset, load_json_file

    if data_path.is_dir():
        print(f"Chargement du dossier : {data_path}")
        df = load_dataset(data_path)
        n_files = df["source_file"].nunique()
        print(f"  {len(df)} records — {df['patient'].nunique()} patients — {n_files} fichier(s)")
    else:
        print(f"Chargement du fichier : {data_path}")
        records = load_json_file(data_path)
        df = pd.DataFrame(records)
        df = df.sort_values(["patient", "visit_date"]).reset_index(drop=True)
        print(f"  {len(df)} records — {df['patient'].nunique()} patients")

    return df, False


def build_features(df: pd.DataFrame, mode: str):
    from src.features import build_delta_features, build_feature_matrix, preprocess

    print(f"\nConstruction des features ({mode})...")

    if mode == "abs":
        feature_df, feature_cols = build_feature_matrix(df)
        X, scaler, imputer = preprocess(feature_df, fit=True)
        df_pipeline = df
        print(f"  Shape matrice : {X.shape} — {len(feature_cols)} features")
        return X, df_pipeline, feature_df, scaler, imputer, None

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
        return X, df_pipeline, feature_df, scaler, imputer, None
    else:
        delta_filtered = delta_df[delta_mask].reset_index(drop=True)
        X, scaler, imputer = preprocess(delta_filtered, fit=True)
        df_pipeline = df[delta_mask].reset_index(drop=True)
        feature_df = feature_df[delta_mask].reset_index(drop=True)

    print(f"  Shape matrice : {X.shape}")
    return X, df_pipeline, feature_df, scaler, imputer, delta_filtered


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
    loss_history: list,
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

    # Courbe de loss autoencoder
    _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(loss_history) + 1), loss_history, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Courbe d'entraînement — Autoencoder")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "loss_curve.png", dpi=120)
    plt.close()

    print(f"Figures sauvegardées → {figures_dir}/")


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)

    if not data_path.exists():
        print(f"Erreur : chemin introuvable → {data_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else Path("results") / args.mode

    df_raw, already_clean = load_data(data_path)

    if already_clean:
        df = df_raw
    else:
        from src.cleaning import clean_dataset, cleaning_report
        df, rejected_df = clean_dataset(df_raw)
        cleaning_report(df_raw, df, rejected_df)

    X, df_pipeline, feature_df, scaler, imputer, delta_df = build_features(df, args.mode)

    print("\nLancement du pipeline non supervisé...")
    from src.models.unsupervised import run_unsupervised_pipeline

    scores_df, if_model, ae_model, loss_history = run_unsupervised_pipeline(
        X,
        contamination=args.contamination,
        ae_epochs=args.epochs,
        device=args.device,
    )

    # Règle NIHL : encoche ≥ 10 dB à 3, 4 ou 6 kHz (Coles et al. 2000)
    # Comparaison directe aux fréquences de référence — capte aussi les audiogrammes
    # sans récupération à 8 kHz que la méthode dérivée manque.
    _nihl_thr = 10
    nihl_flag = (
        (feature_df["notch_3k_L"].fillna(0) > _nihl_thr) |
        (feature_df["notch_4k_L"].fillna(0) > _nihl_thr) |
        (feature_df["notch_6k_L"].fillna(0) > _nihl_thr) |
        (feature_df["notch_3k_R"].fillna(0) > _nihl_thr) |
        (feature_df["notch_4k_R"].fillna(0) > _nihl_thr) |
        (feature_df["notch_6k_R"].fillna(0) > _nihl_thr)
    ).astype(int)
    nihl_flag.index = scores_df.index

    # Règle Ménière : PTA BF corrigé > 25 dB (résidu vs norme âge/genre)
    low_L = feature_df[["L_250", "L_500", "L_1000"]].mean(axis=1)
    low_R = feature_df[["R_250", "R_500", "R_1000"]].mean(axis=1)
    meniere_flag = ((low_L.fillna(0) > 25) | (low_R.fillna(0) > 25)).astype(int)
    meniere_flag.index = scores_df.index

    # Règle STS OSHA : shift ≥ 10 dB en moyenne 2/3/4 kHz vs Baseline (mode delta uniquement)
    if delta_df is not None and "has_sts_L" in delta_df.columns:
        sts_flag = (
            (delta_df["has_sts_L"].fillna(0) == 1) |
            (delta_df["has_sts_R"].fillna(0) == 1)
        ).astype(int)
        sts_flag.index = scores_df.index
    else:
        sts_flag = pd.Series(0, index=scores_df.index)

    scores_df["nihl_flag"]    = nihl_flag
    scores_df["meniere_flag"] = meniere_flag
    scores_df["sts_flag"]     = sts_flag
    scores_df["anomaly_final"] = (
        (scores_df["anomaly_consensus"] == 1) |
        (scores_df["nihl_flag"] == 1) |
        (scores_df["meniere_flag"] == 1) |
        (scores_df["sts_flag"] == 1)
    ).astype(int)

    print("\n=== Résultats ===")
    from src.evaluate import summary_report
    summary_report(df_pipeline, scores_df)

    from src.iso7029_validation import (
        compute_iso7029_residuals,
        compute_precision_at_iso7029,
        print_iso7029_report,
    )
    residuals_df = compute_iso7029_residuals(df_pipeline)
    iso_metrics = compute_precision_at_iso7029(residuals_df, scores_df)
    print_iso7029_report(iso_metrics)

    save_outputs(output_dir, scores_df, if_model, ae_model, scaler, imputer)

    if not args.no_plots:
        print("\nGénération des figures...")
        generate_plots(output_dir, df_pipeline, scores_df, X, loss_history)

        from src.iso7029_validation import plot_iso7029_validation
        plot_iso7029_validation(residuals_df, scores_df, output_dir)

    print("\nTerminé.")


if __name__ == "__main__":
    main()
