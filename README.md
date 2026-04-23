# Odyo Anomaly Detection — Détection d'anomalies sur audiogrammes

Projet de Machine Learning développé pour **Odyo Soft** dans le cadre d'un stage EIC Co-op.

L'objectif est de détecter automatiquement des audiogrammes anormaux à partir d'exports de la base de données Odyo, en exploitant à la fois les seuils auditifs mesurés et les métadonnées associées.

---

## Contexte métier

### Qu'est-ce qu'un audiogramme ?

Un **audiogramme** est le résultat d'un test auditif (audiométrie tonale). Il mesure le **seuil d'audition** d'un patient, c'est-à-dire le niveau sonore minimal qu'il peut entendre, à différentes **fréquences** (de 250 Hz aux graves jusqu'à 8000 Hz dans les aigus), pour **chaque oreille** (gauche et droite).

Les seuils sont exprimés en **décibels HL (Hearing Level)**. Plus la valeur est élevée, plus la perte auditive est importante :

| Seuil moyen (PTA) | Interprétation |
|---|---|
| 0 – 25 dB | Audition normale |
| 26 – 40 dB | Perte légère |
| 41 – 55 dB | Perte modérée |
| 56 – 70 dB | Perte modérément sévère |
| > 70 dB | Perte sévère à profonde |

### Fréquences standard testées

```
250 Hz  →  500 Hz  →  1000 Hz  →  2000 Hz  →  4000 Hz  →  8000 Hz
(graves)                                                    (aigus)
```

### Types de pertes auditives

- **Surdité de perception (neurosensorielle)** : atteinte de la cochlée ou du nerf auditif. Courbe descendante en hautes fréquences.
- **Surdité de transmission (conductive)** : problème mécanique (oreille moyenne). Seuils osseux normaux, seuils aériens élevés.
- **Surdité mixte** : combinaison des deux.

### Qu'est-ce qu'une anomalie ici ?

Une anomalie peut être :
- Un audiogramme avec une configuration inhabituelle (asymétrie marquée, chute abrupte)
- Un résultat incohérent (seuils physiologiquement impossibles)
- Un pattern clinique rare ou sévère
- Une erreur de saisie ou de mesure

---

## Format des données

Les données proviennent d'un export **MongoDB** au format JSON. Chaque fichier JSON contient un ou plusieurs **records**, où chaque record = un rapport d'audiométrie.

### Structure d'un record

```json
{
  "_id": { "$oid": "..." },           // Identifiant unique du rapport
  "patient": "hash_anonymisé",        // Patient (anonymisé)
  "category": 0,                      // Potentiel label de classification
  "version": 1,
  "visitDate": { "$date": "..." },    // Date de la consultation
  "data": {
    "type": 3,                        // Type de test (3 = audiogramme)
    "audiogramme": {
      "evaluationMode": 0,
      "hearingLine": "25",            // Seuil de référence en dB
      "dots": {
        "left": [                     // Points audiogramme OG
          [dB, fréquence_Hz, transducteur, ?, masked],
          [35, 500, 1, 0, false],     // Ex : 35 dB à 500 Hz, non masqué
          ...
        ],
        "right": [ ... ]              // Points audiogramme OD
      }
    },
    "divers": {
      "testValidity": 0,              // 0 = test valide
      "transducers": [1],             // Type de transducteur (1 = aérien)
      "reportDate": "..."
    }
  }
}
```

### Format d'un point (`dots`)

Chaque point est un tableau de 5 valeurs :

| Index | Valeur | Signification |
|---|---|---|
| 0 | `dB` | Seuil auditif mesuré (décibels HL) |
| 1 | `fréquence` | Fréquence testée en Hz (ex : 500, 1000...) |
| 2 | `transducteur` | Type : 1 = aérien (casque), 2 = osseux (vibrateur) |
| 3 | `?` | Réservé (actuellement 0) |
| 4 | `masked` | `true` si la mesure a été faite avec masquage contralateral |

**Important :** les fréquences testées ne sont pas toujours les mêmes entre records (ex : 375 Hz, 1500 Hz, 3000 Hz). Une étape d'**interpolation** est nécessaire pour aligner tous les records sur les fréquences standard.

---

## Architecture du projet

```
odyo_anomaly/
│
├── data/
│   └── raw/                  # Fichiers JSON bruts (exports MongoDB)
│                             # → Placer vos fichiers .json ici
│
├── notebooks/
│   └── 01_exploration.ipynb  # Notebook d'exploration et tests
│
├── src/
│   ├── __init__.py
│   ├── loader.py             # Chargement et parsing des JSON MongoDB
│   ├── features.py           # Feature engineering (interpolation, PTA, etc.)
│   ├── evaluate.py           # Visualisation et métriques
│   └── models/
│       ├── __init__.py
│       ├── unsupervised.py   # Isolation Forest + Autoencoder + PCA
│       └── supervised.py     # XGBoost + Random Forest + One-Class SVM
│
├── requirements.txt
└── README.md
```

### Description des fichiers sources

#### `src/loader.py`
Parse les fichiers JSON MongoDB :
- Gère le format MongoDB extended JSON (`$oid`, `$date`)
- Filtre les records supprimés (`isDeleted=True`) et invalides (`testValidity≠0`)
- Extrait les `dots` (points audiogramme) en dictionnaires `{fréquence_Hz: dB}`
- Fonctions principales : `load_json_file()`, `load_dataset()`

#### `src/features.py`
Construit les vecteurs numériques pour le ML :
- **Interpolation** linéaire des seuils vers les 6 fréquences standard (250–8000 Hz)
- **PTA** (Pure Tone Average) : moyenne 500+1000+2000+4000 Hz par oreille
- **Chute haute fréquence** : différence 4000 Hz → 8000 Hz (indicateur presbyacousie)
- **Asymétrie inter-oreilles** : moyenne |OG - OD| sur les fréquences communes
- **Imputation** des valeurs manquantes par la médiane du dataset
- **Standardisation** (moyenne=0, écart-type=1) pour les algorithmes sensibles à l'échelle

#### `src/models/unsupervised.py`
Pipeline non supervisé (ne nécessite pas de labels) :
- `IsolationForest` : détecte les points aberrants par isolation aléatoire
- `Autoencoder` (PyTorch) : apprend la distribution normale, score = erreur de reconstruction
- `PCA reconstruction error` : baseline rapide par compression linéaire
- **Consensus** : flagge une anomalie si au moins 2 méthodes sur 3 sont d'accord

#### `src/models/supervised.py`
Pipeline supervisé (nécessite des labels dans `category`) :
- `XGBoost` : boosting de gradient, gestion du déséquilibre par `scale_pos_weight`
- `RandomForest` : bagging, interprétable via `feature_importances_`
- `One-Class SVM` : entraîné uniquement sur les cas normaux (semi-supervisé)
- Validation croisée stratifiée (5-fold) avec Precision, Recall, F1, ROC-AUC

#### `src/evaluate.py`
Visualisation des résultats :
- Histogramme des scores d'anomalie avec seuil
- Audiogramme standard (courbe OG × / courbe OD ○, axe Y inversé)
- Top-N audiogrammes les plus anormaux
- Projection UMAP 2D colorée par score ou label
- Matrice de confusion et courbe ROC (mode supervisé)
- Importance des features (mode supervisé)

---

## Installation

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Installer les dépendances
pip install -r requirements.txt
```

---

## Utilisation

### 1. Placer les données

Copier vos fichiers JSON dans `data/raw/` :
```
data/raw/
├── export_janvier_2025.json
├── export_fevrier_2025.json
└── ...
```

Chaque fichier peut contenir un seul record ou un tableau de records.

### 2. Lancer le notebook d'exploration

```bash
cd notebooks
jupyter notebook 01_exploration.ipynb
```

Le notebook guide à travers toutes les étapes : chargement, exploration, feature engineering, puis les deux pipelines ML.

### 3. Utilisation programmatique

```python
from src.loader import load_dataset
from src.features import build_feature_matrix, preprocess
from src.models.unsupervised import run_unsupervised_pipeline
from src.evaluate import summary_report, plot_umap

# Charger les données
df = load_dataset('data/raw/')

# Construire les features
feature_df, feature_cols = build_feature_matrix(df)
X, scaler, imputer = preprocess(feature_df, fit=True)

# Pipeline non supervisé
scores_df, if_model, ae_model = run_unsupervised_pipeline(X, contamination=0.05)

# Résumé
summary_report(df, scores_df)

# Visualisation UMAP
plot_umap(X, scores_df['reconstruction_error'].values)
```

---

## Pipeline non supervisé — Détail

Utilisé quand aucun label de pathologie n'est disponible.

### Isolation Forest

**Principe :** construit des arbres de décision aléatoires. Un point est anormal s'il est facilement isolé (chemin court dans l'arbre). Fonctionne bien sur des données tabulaires.

**Paramètre clé :** `contamination` = proportion estimée d'anomalies dans le dataset (défaut : 5%). À ajuster selon le contexte clinique.

**Sortie :**
- `anomaly_score_if` : score brut (plus négatif = plus anormal)
- `anomaly_flag_if` : 1 = anomalie, 0 = normal

### Autoencoder (PyTorch)

**Principe :** réseau de neurones entraîné à compresser puis reconstruire les audiogrammes. Après entraînement sur la majorité des données (supposées normales), il reconstruit mal les cas atypiques — leur erreur de reconstruction est donc élevée.

**Architecture :**
```
Input(N features) → Linear(16) → ReLU → Linear(6) → ReLU   ← encodeur
                 → Linear(16) → ReLU → Linear(N)            ← décodeur
```

**Seuil :** percentile 95 des erreurs de reconstruction sur l'ensemble d'entraînement.

**Sortie :**
- `reconstruction_error` : MSE par sample
- `anomaly_flag_ae` : 1 si > percentile 95

### Consensus

Un record est marqué **anomalie de consensus** si au moins 2 méthodes sur 3 le signalent. Cette approche réduit les faux positifs de chaque méthode individuelle.

---

## Pipeline supervisé — Détail

Utilisé quand le champ `category` contient des labels de pathologie confirmés.

### Prérequis

Il faut d'abord vérifier que `category` encode bien une information utile :
```python
df['category'].value_counts()
```

Si `category` distingue cas normaux (0) et anomalies (1 ou plus), binariser :
```python
y = (df['category'] != 0).astype(int)
```

### XGBoost (recommandé)

Gradient boosting : entraîne des arbres séquentiellement, chacun corrigeant les erreurs du précédent. Gère le déséquilibre de classes via `scale_pos_weight = n_normaux / n_anomalies`.

### Random Forest

Bagging d'arbres de décision. Plus interprétable via `feature_importances_` — utile pour comprendre quelles fréquences sont les plus discriminantes cliniquement.

### Évaluation

La validation croisée **stratifiée** (5-fold) garantit que chaque fold contient une proportion représentative d'anomalies :

| Métrique | Description |
|---|---|
| **Precision** | Parmi les records flaggés anomalie, quelle fraction l'est vraiment ? |
| **Recall** | Parmi les vraies anomalies, quelle fraction a été détectée ? |
| **F1** | Moyenne harmonique Precision/Recall |
| **ROC-AUC** | Capacité à discriminer normal/anomalie (1.0 = parfait) |

**En contexte médical, le Recall (sensibilité) est souvent prioritaire** : mieux vaut un faux positif qu'une anomalie manquée.

---

## Features construites

| Feature | Description | Calcul |
|---|---|---|
| `L_250` … `L_8000` | Seuils OG aux fréquences standard | Interpolation linéaire |
| `R_250` … `R_8000` | Seuils OD aux fréquences standard | Interpolation linéaire |
| `PTA_L` | Pure Tone Average OG | Moyenne(500, 1000, 2000, 4000 Hz) |
| `PTA_R` | Pure Tone Average OD | Moyenne(500, 1000, 2000, 4000 Hz) |
| `high_freq_drop_L` | Chute HF oreille gauche | dB(8000) − dB(4000) |
| `high_freq_drop_R` | Chute HF oreille droite | dB(8000) − dB(4000) |
| `asymmetry_mean` | Asymétrie inter-oreilles | Moyenne(\|OG − OD\|) |
| `hearing_line` | Seuil de référence du rapport | Champ `hearingLine` |
| `evaluation_mode` | Mode d'évaluation | Champ `evaluationMode` |
| `n_freqs_left` | Nb fréquences testées OG | `len(dots.left)` |
| `n_freqs_right` | Nb fréquences testées OD | `len(dots.right)` |

**Total : 19 features** (6+6 seuils + 7 features dérivées)

---

## Questions ouvertes

- **`category`** : quelles valeurs possibles ? Est-ce un label de pathologie ou de type de test ?
- **Âge / genre** : les métadonnées patient sont-elles dans un fichier séparé ? (le champ `patient` est un hash anonymisé dans ce format)
- **Transducteurs osseux** : les mesures avec `transducer=2` doivent-elles être traitées séparément ?
- **Volume final** : combien de records au total ? (impacte le choix des hyperparamètres)

---

## Dépendances

| Librairie | Usage |
|---|---|
| `pandas`, `numpy` | Manipulation et calcul matriciel |
| `scipy` | Interpolation des seuils |
| `scikit-learn` | Isolation Forest, PCA, preprocessing, métriques |
| `torch` | Autoencoder (détection non supervisée deep learning) |
| `xgboost` | Classification supervisée |
| `matplotlib`, `seaborn` | Visualisations |
| `umap-learn` | Réduction dimensionnelle pour visualisation |
| `jupyter` | Notebooks d'exploration |
