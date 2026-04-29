"""
Pipeline de détection d'anomalies non supervisée.

Deux approches complémentaires :
1. Isolation Forest (scikit-learn) — rapide, interprétable, bon pour données tabulaires
2. Autoencoder (PyTorch) — capture les patterns non linéaires, score = erreur de reconstruction
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA


# ─── Isolation Forest ────────────────────────────────────────────────────────

def train_isolation_forest(
    X: np.ndarray,
    contamination: float = 0.05,
    random_state: int = 42,
) -> IsolationForest:
    """
    Entraîne un Isolation Forest.

    contamination : proportion estimée d'anomalies dans le dataset.
    Retourne le modèle ajusté.
    """
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)
    return model


def score_isolation_forest(model: IsolationForest, X: np.ndarray) -> pd.DataFrame:
    """
    Calcule les scores d'anomalie pour chaque sample.

    Retourne un DataFrame avec :
    - anomaly_score : score brut (plus négatif = plus anormal)
    - anomaly_flag  : 1 = anomalie, 0 = normal (seuil calculé par contamination)
    """
    raw_scores = model.score_samples(X)
    flags = (model.predict(X) == -1).astype(int)
    return pd.DataFrame({"anomaly_score_if": raw_scores, "anomaly_flag_if": flags})


# ─── Autoencoder ─────────────────────────────────────────────────────────────

class Autoencoder(nn.Module):
    """
    Autoencoder fully-connected pour données tabulaires.

    Architecture : Input(n) → hidden1 → latent → hidden1 → Output(n)
    L'erreur de reconstruction (MSE) sert de score d'anomalie.
    Un audiogramme anormal aura une forte erreur car le modèle
    a appris uniquement la distribution des cas normaux.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 16, latent_dim: int = 6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def train_autoencoder(
    X: np.ndarray,
    hidden_dim: int = 16,
    latent_dim: int = 6,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
) -> tuple["Autoencoder", list[float]]:
    """
    Entraîne l'Autoencoder sur les données X (supposées majoritairement normales).

    Retourne le modèle entraîné et l'historique de loss (une valeur par epoch).
    """
    input_dim = X.shape[1]
    model = Autoencoder(input_dim, hidden_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_history = []
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        loss_history.append(avg)
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs} — loss: {avg:.4f}")

    return model, loss_history


def score_autoencoder(model: Autoencoder, X: np.ndarray, device: str = "cpu") -> pd.DataFrame:
    """
    Calcule l'erreur de reconstruction pour chaque sample.

    Retourne un DataFrame avec :
    - reconstruction_error : MSE par sample (plus élevé = plus anormal)
    - anomaly_flag_ae      : 1 si reconstruction_error > percentile 95
    """
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        reconstructed = model(X_tensor).cpu().numpy()

    errors = np.mean((X - reconstructed) ** 2, axis=1)
    threshold = np.percentile(errors, 95)
    flags = (errors > threshold).astype(int)

    return pd.DataFrame({"reconstruction_error": errors, "anomaly_flag_ae": flags})


# ─── PCA baseline ─────────────────────────────────────────────────────────────

def score_pca_reconstruction(X: np.ndarray, n_components: int = 5) -> pd.DataFrame:
    """
    Baseline : détecte les anomalies par erreur de reconstruction PCA.

    Les samples mal reconstruits (hors de l'espace principal) sont suspects.
    """
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_reduced)
    errors = np.mean((X - X_reconstructed) ** 2, axis=1)
    threshold = np.percentile(errors, 95)
    flags = (errors > threshold).astype(int)
    return pd.DataFrame({"pca_reconstruction_error": errors, "anomaly_flag_pca": flags})


# ─── Pipeline complet ─────────────────────────────────────────────────────────

def run_unsupervised_pipeline(
    X: np.ndarray,
    contamination: float = 0.05,
    ae_epochs: int = 100,
    device: str = "cpu",
    feature_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, IsolationForest, Autoencoder]:
    """
    Lance les trois méthodes non supervisées et combine leurs scores.

    Retourne :
    - scores_df     : DataFrame avec tous les scores et flags
    - if_model      : Isolation Forest entraîné
    - ae_model      : Autoencoder entraîné
    """
    print("→ Isolation Forest...")
    if_model = train_isolation_forest(X, contamination=contamination)
    if_scores = score_isolation_forest(if_model, X)

    print("→ Autoencoder...")
    ae_model, loss_history = train_autoencoder(X, epochs=ae_epochs, device=device)
    ae_scores = score_autoencoder(ae_model, X, device=device)

    print("→ PCA baseline...")
    pca_scores = score_pca_reconstruction(X)

    scores_df = pd.concat([if_scores, ae_scores, pca_scores], axis=1)

    # Consensus : anomalie si au moins 2 méthodes sur 3 sont d'accord
    scores_df["anomaly_consensus"] = (
        scores_df["anomaly_flag_if"]
        + scores_df["anomaly_flag_ae"]
        + scores_df["anomaly_flag_pca"]
        >= 2
    ).astype(int)

    # Règle clinique NIHL : creux > 15 dB entre 2–8 kHz (dérivée discrète)
    if feature_df is not None:
        scores_df["nihl_flag"] = (
            (feature_df["notch_depth_L"].fillna(0) > 15) |
            (feature_df["notch_depth_R"].fillna(0) > 15)
        ).astype(int).values

    return scores_df, if_model, ae_model, loss_history
