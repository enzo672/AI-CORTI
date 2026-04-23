"""
Pipeline de détection d'anomalies supervisée.

Utilisé lorsque les labels sont disponibles (champ `category` confirmé).

Deux modèles proposés :
1. Random Forest — interprétable, robuste, bon point de départ
2. XGBoost — généralement plus performant, gère mieux le déséquilibre de classes
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import OneClassSVM
import xgboost as xgb


# ─── Random Forest ───────────────────────────────────────────────────────────

def train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 200,
    class_weight: str = "balanced",
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Entraîne un Random Forest avec gestion du déséquilibre de classes.

    class_weight='balanced' compense automatiquement si les anomalies
    sont rares (ce qui est souvent le cas).
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


# ─── XGBoost ──────────────────────────────────────────────────────────────────

def train_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
) -> xgb.XGBClassifier:
    """
    Entraîne un XGBoost avec `scale_pos_weight` pour gérer le déséquilibre.

    scale_pos_weight = n_négatifs / n_positifs : donne plus de poids
    aux anomalies (classe minoritaire) pendant l'entraînement.
    """
    n_neg = int(np.sum(y == 0))
    n_pos = int(np.sum(y == 1))
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=200,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


# ─── One-Class SVM (semi-supervisé) ──────────────────────────────────────────

def train_one_class_svm(
    X_normal: np.ndarray,
    nu: float = 0.05,
) -> OneClassSVM:
    """
    Entraîne un One-Class SVM uniquement sur les échantillons normaux.

    Utile quand seuls les cas normaux sont labelisés.
    nu : proportion maximale d'erreurs de formation (similaire à contamination).
    """
    model = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
    model.fit(X_normal)
    return model


def score_one_class_svm(model: OneClassSVM, X: np.ndarray) -> pd.DataFrame:
    """
    Prédit les anomalies avec le One-Class SVM.
    Retourne anomaly_flag_ocsvm : 1 = anomalie, 0 = normal.
    """
    predictions = model.predict(X)
    flags = (predictions == -1).astype(int)
    scores = model.score_samples(X)
    return pd.DataFrame({"ocsvm_score": scores, "anomaly_flag_ocsvm": flags})


# ─── Évaluation croisée ──────────────────────────────────────────────────────

def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Évalue un modèle par validation croisée stratifiée.

    Retourne un DataFrame résumant precision, recall, F1 et ROC-AUC
    pour chaque fold.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = ["precision", "recall", "f1", "roc_auc"]
    results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)

    summary = pd.DataFrame({
        "fold": range(1, n_splits + 1),
        "precision": results["test_precision"],
        "recall": results["test_recall"],
        "f1": results["test_f1"],
        "roc_auc": results["test_roc_auc"],
    })
    print(summary.to_string(index=False))
    print(f"\nMoyennes → Precision: {summary['precision'].mean():.3f} | "
          f"Recall: {summary['recall'].mean():.3f} | "
          f"F1: {summary['f1'].mean():.3f} | "
          f"ROC-AUC: {summary['roc_auc'].mean():.3f}")
    return summary


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Évalue un modèle entraîné sur un jeu de test.

    Retourne un dict de métriques.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)

    print(classification_report(y_test, y_pred, target_names=["normal", "anomalie"]))
    return metrics


# ─── Pipeline complet ─────────────────────────────────────────────────────────

def run_supervised_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "xgboost",
) -> tuple:
    """
    Lance le pipeline supervisé complet avec validation croisée.

    model_type : 'xgboost' ou 'random_forest'

    Retourne (model, cv_results).
    """
    print(f"\n=== Pipeline supervisé — {model_type} ===")
    print(f"Distribution des labels : {dict(zip(*np.unique(y, return_counts=True)))}\n")

    if model_type == "xgboost":
        model = train_xgboost(X, y)
    else:
        model = train_random_forest(X, y)

    print(f"→ Validation croisée ({5} folds)...")
    cv_results = cross_validate_model(model, X, y)

    model.fit(X, y)
    return model, cv_results
