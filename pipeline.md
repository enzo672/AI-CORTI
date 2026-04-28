# Pipeline AI-CORTI — Documentation détaillée

> Détection d'anomalies non supervisée sur audiogrammes exportés depuis MongoDB (Odyo).

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Étape 1 — Chargement des données](#2-étape-1--chargement-des-données)
3. [Étape 2 — Feature engineering](#3-étape-2--feature-engineering)
4. [Étape 3 — Modèles de détection](#4-étape-3--modèles-de-détection)
5. [Étape 4 — Évaluation & figures](#5-étape-4--évaluation--figures)
6. [Fichiers de sortie](#6-fichiers-de-sortie)
7. [Lancer le pipeline](#7-lancer-le-pipeline)

---

## 1. Vue d'ensemble

L'objectif est de détecter automatiquement les **audiogrammes atypiques** sans labels préétablis (pas besoin de dire "ceci est normal / anormal" manuellement). C'est du **machine learning non supervisé**.

### Flux complet

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DONNÉES D'ENTRÉE                           │
│             Fichiers JSON exportés depuis MongoDB Odyo              │
│          (1 fichier = 1 rapport d'audiométrie complet)              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ÉTAPE 1 — LOADER (src/loader.py)                │
│   • Parse les points d'audiogramme OG/OD                           │
│   • Filtre les records invalides / supprimés                        │
│   • Identifie Baseline / Periodic / Depart                          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  DataFrame : 1 ligne = 1 rapport
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              ÉTAPE 2 — FEATURE ENGINEERING (src/features.py)       │
│   • Interpole les seuils aux 6 fréquences standard  (250Hz, 500Hz, 1kHz, 2kHz, 4kHz, 8kHz)                │
│   • Calcule PTA, asymétrie, chute HF, STS...                       │
│   • Impute les NaN (médiane) + standardise (z-score)               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  Matrice X : (N patients × 15 features)
                               ▼
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  Isolation   │  │ Autoencoder  │  │     PCA      │
    │   Forest     │  │  (PyTorch)   │  │  baseline    │
    └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
           └─────────────────┼─────────────────┘
                             │  Scores d'anomalie par patient
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CONSENSUS (≥ 2 méthodes / 3)                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
         scores.csv       Modèles .pt      4 Figures
                          .joblib          (results/figures/)
```

---

## 2. Étape 1 — Chargement des données

**Fichier :** [src/loader.py](src/loader.py)

### Format d'un point d'audiogramme

Chaque point mesuré sur l'audiogramme est encodé comme un tableau de 5 valeurs dans le JSON MongoDB :

```
[dB, fréquence_Hz, dot_category, réservé, no_response]

Exemple : [-10.0, 1000, 1, 0, false]
           ↑       ↑    ↑        ↑
           Seuil   1kHz rapport  réponse OK
                        courant
```

| Champ | Valeur | Signification |
|---|---|---|
| `dot_category` | `1` | Point du rapport **courant** → conservé |
| `dot_category` | `3` | Point du rapport **précédent** (gris UI) → ignoré |
| `no_response` | `true` | Patient n'a pas répondu → point exclu |

### Catégories de visite

```
visit_category  0 = Baseline  → audiogramme de référence initiale du patient
                1 = Periodic  → suivi régulier
                2 = Depart    → bilan de sortie
```

### Filtres appliqués

```
JSON brut
    │
    ├── isDeleted == true        → ❌ ignoré
    ├── testValidity != 0        → ❌ invalide (bilan non fiable)
    ├── dot_category == 3        → ❌ points du rapport précédent
    ├── no_response == true      → ❌ pas de réponse patient
    └── record valide            → ✅ conservé dans le DataFrame
```

### Résultat

Un `pd.DataFrame` où chaque ligne est un rapport valide, avec notamment :

| Colonne | Description |
|---|---|
| `record_id` | Identifiant MongoDB |
| `patient` | Hash du patient |
| `visit_category` | 0 / 1 / 2 |
| `visit_date` | Date de la visite |
| `dots_left` | Dict `{fréquence: dB}` oreille gauche |
| `dots_right` | Dict `{fréquence: dB}` oreille droite |

---

## 3. Étape 2 — Feature engineering

**Fichier :** [src/features.py](src/features.py)

Un modèle ML ne comprend pas un audiogramme brut. Il faut le transformer en **vecteur de nombres**.

### 3.1 Interpolation aux fréquences standard

Les audiogrammes ne sont pas toujours mesurés aux mêmes fréquences. On interpole linéairement pour ramener tous les profils aux **6 fréquences standard** :

```
Fréquences standard : 250 Hz, 500 Hz, 1 kHz, 2 kHz, 4 kHz, 8 kHz

Exemple : patient mesuré à 500, 1k, 2k, 4k Hz
          → interpolation à 250 Hz et 8 kHz
          → NaN si hors plage (pas d'extrapolation)
```

### 3.2 Features calculées (mode `abs`)

Pour chaque rapport on obtient **15 features** :

```
┌─────────────────────────────────────────────────────────────────┐
│ SEUILS PAR FRÉQUENCE (12 features)                             │
│  L_250   L_500   L_1000   L_2000   L_4000   L_8000  ← OG (dB) │
│  R_250   R_500   R_1000   R_2000   R_4000   R_8000  ← OD (dB) │
├─────────────────────────────────────────────────────────────────┤
│ DÉRIVÉES CLINIQUES (5 features)                                │
│  PTA_L / PTA_R          → Pure Tone Average (500/1k/2k/4k Hz)  │
│  high_freq_drop_L/R     → Seuil 8kHz − Seuil 4kHz             │
│  asymmetry_mean         → Moyenne |OG − OD| sur toutes freqs   │
├─────────────────────────────────────────────────────────────────┤
│ MÉTADONNÉES (3 features)                                        │
│  hearing_line, evaluation_mode, n_freqs_left/right             │
└─────────────────────────────────────────────────────────────────┘
```

**PTA (Pure Tone Average)** — indicateur standard de la perte auditive globale :
```
PTA = moyenne(500 Hz, 1000 Hz, 2000 Hz, 4000 Hz)

  < 25 dB  → audition normale
  25-40 dB → perte légère
  40-70 dB → perte moyenne
  > 70 dB  → perte sévère à profonde
```

**Chute haute fréquence** — signe typique d'un traumatisme sonore ou presbyacousie :
```
high_freq_drop = seuil_8kHz − seuil_4kHz

  Valeur positive élevée → grosse chute dans les aigus (encoche 4kHz typique)
```

**Asymétrie** — une différence importante OG/OD est cliniquement suspecte :
```
asymmetry = moyenne( |OG_f − OD_f| ) pour chaque fréquence f

  > 15-20 dB → asymétrie significative, potentiellement pathologique
```

### 3.3 Features delta (mode `delta`, optionnel)

Au lieu des valeurs absolues, on mesure l'**évolution par rapport au Baseline** du même patient. Utile pour détecter une dégradation progressive.

```
Periodic/Depart
      │
      │  delta_L_f = seuil_courant(f) − seuil_baseline(f)
      │
      ▼
  delta_L_250, delta_L_500, ..., delta_R_8000
  delta_PTA_L, delta_PTA_R
  sts_L, sts_R   ← Standard Threshold Shift (OSHA : 2k/3k/4k Hz)
  has_sts_L/R    ← flag : STS ≥ 10 dB (cliniquement significatif)
```

> Le **STS OSHA** est un critère réglementaire : si la moyenne aux fréquences 2000/3000/4000 Hz augmente de ≥ 10 dB vs la Baseline, c'est une alerte de détérioration auditive professionnelle.

### 3.4 Préprocessing

```
Feature matrix brute (avec NaN)
         │
         ▼
   SimpleImputer          → remplace chaque NaN par la médiane de la colonne
         │                   (ex: fréquence 250 Hz non mesurée → médiane du dataset)
         ▼
   StandardScaler         → centre et réduit chaque feature
                             nouvelle_valeur = (valeur − moyenne) / écart_type
                             → toutes les features ont moyenne=0, std=1
```

Sans standardisation, le modèle serait biaisé par l'ordre de grandeur : un seuil à 80 dB "dominerait" une asymétrie à 5 dB.

---

## 4. Étape 3 — Modèles de détection

**Fichier :** [src/models/unsupervised.py](src/models/unsupervised.py)

### 4.1 Isolation Forest

**Principe :** un profil anormal est plus facile à isoler des autres qu'un profil courant. L'algorithme construit 200 arbres de décision aléatoires. Il mesure combien de coupures sont nécessaires pour isoler chaque patient.

```
Profil NORMAL    → beaucoup de coupures nécessaires → score proche de 0
Profil ANORMAL   → peu de coupures nécessaires      → score très négatif

                 ─────────────────────────────────────▶
Score :       -0.57    -0.50    -0.45    -0.43
               ↑                           ↑
            Anormal                      Normal
           (patient #9)              (patients courants)
```

**Paramètre clé :** `contamination=0.05` → on lui dit "environ 5% des patients sont anormaux". Cela fixe le seuil de décision.

**Sortie :**
| Colonne | Description |
|---|---|
| `anomaly_score_if` | Score brut (plus négatif = plus anormal) |
| `anomaly_flag_if` | `1` = signalé anormal, `0` = normal |

---

### 4.2 Autoencoder (réseau de neurones)

**Principe :** on entraîne un réseau à **compresser** puis **reconstruire** les audiogrammes. Un profil typique sera bien reconstruit. Un profil inhabituel sera mal reconstruit car le réseau n'a jamais appris ce type de pattern.

**Architecture :**
```
           ENCODEUR                    DÉCODEUR
              │                           │
 Input        │   Compression             │   Reconstruction
 (15 valeurs) │                           │   (15 valeurs)
      │       ▼                           ▼       │
      │   ┌──────┐    ┌──────┐    ┌──────┐   │
      └──▶│  16  │───▶│  6   │───▶│  16  │───┘
          │neurones│  │latent│   │neurones│
          └──────┘    └──────┘    └──────┘
            ReLU        ReLU        ReLU

          ◀──────── 100 epochs ────────▶
               (optimisation Adam)
```

Le **vecteur latent de 6 valeurs** est une "empreinte compressée" de l'audiogramme. Le réseau apprend à encoder les audiogrammes typiques efficacement. Un audiogramme atypique ne rentre pas bien dans ce moule → erreur de reconstruction élevée.

**Score :** MSE (erreur quadratique moyenne) entre entrée et reconstruction :
```
reconstruction_error = moyenne( (valeur_entrée − valeur_reconstruite)² )

  Proche de 0   → profil bien reconnu par le réseau → normal
  Élevé (>0.5)  → profil mal reconstruit → suspect
```

**Seuil de détection :** 95ème percentile des erreurs → les 5% avec la plus grosse erreur sont flaggés.

**Sortie :**
| Colonne | Description |
|---|---|
| `reconstruction_error` | Erreur MSE (plus grande = plus anormal) |
| `anomaly_flag_ae` | `1` si erreur > percentile 95 |

---

### 4.3 PCA baseline

**Principe :** la PCA (Analyse en Composantes Principales) trouve les "directions" qui expliquent le plus la variance du dataset. On réduit à 5 composantes puis on reconstruit. Un profil bien "dans la norme" du groupe sera bien reconstruit.

```
  15 features originales
        │
        ▼  PCA fit_transform
  5 composantes principales (résumé de la variance principale)
        │
        ▼  PCA inverse_transform
  15 features reconstruites
        │
        ▼
  Erreur = distance entre original et reconstruit
```

C'est la méthode la plus simple et la plus rapide. Elle sert de **référence** pour valider que l'Isolation Forest et l'Autoencoder font mieux.

**Sortie :**
| Colonne | Description |
|---|---|
| `pca_reconstruction_error` | Erreur de reconstruction PCA |
| `anomaly_flag_pca` | `1` si erreur > percentile 95 |

---

### 4.4 Consensus

Pour réduire les faux positifs, un patient n'est marqué **vraiment anormal** que si **au moins 2 modèles sur 3** le signalent :

```
anomaly_flag_if  +  anomaly_flag_ae  +  anomaly_flag_pca  ≥  2
      │                   │                   │
      └───────────────────┴───────────────────┘
                          │
                    anomaly_consensus = 1
```

| `anomaly_consensus` | Signification |
|---|---|
| `0` | Pas de consensus → probablement normal |
| `1` | ≥ 2 méthodes d'accord → probablement anormal |

---

## 5. Étape 4 — Évaluation & figures

**Fichier :** [src/evaluate.py](src/evaluate.py)

### Figure 1 — Distribution des scores

![Score distributions](results/figures/score_distributions.png)

Trois histogrammes côte à côte, un par méthode. Chaque barre = combien de patients ont un score dans cet intervalle.

```
Isolation Forest         Autoencoder              PCA baseline
─────────────────        ────────────────         ──────────────
Score : -0.6 → 0        Score : 0 → 1.3+         Score : 0 → 0.11

 Plus négatif            Plus élevé               Plus élevé
 = plus anormal          = plus anormal            = plus anormal
```

Avec beaucoup de données, on verrait une cloche sur la gauche (normaux) et quelques points isolés à droite/à gauche (anomalies). Avec 11 patients, chaque barre = 1-2 patients.

---

### Figure 2 — Top anomalies

![Top anomalies](results/figures/top_anomalies.png)

Les **6 audiogrammes les plus anormaux** selon l'autoencoder, affichés en convention audiométrique standard.

**Comment lire un audiogramme ici :**

```
  0 dB ─────── Ligne du haut : audition parfaite
               ↓
 25 dB ─ ─ ─  Ligne pointillée grise : seuil de normalité
               ↓
 50 dB ─────── Zone de perte modérée
               ↓
120 dB ─────── Ligne du bas : surdité profonde

Axe X  : fréquences (graves 250Hz → aigus 8000Hz)
Croix bleues  : oreille gauche (OG)
Ronds rouges  : oreille droite (OD)
```

**Interprétation des cas visibles :**

| Patient | Score | Pattern observé | Hypothèse clinique |
|---|---|---|---|
| `#0 [Baseline]` | 1.287 | Perte progressive sur les aigus, OG ≈ OD | Presbyacousie / perte neurosensorielle |
| `#9 [Periodic]` | 0.621 | Chute marquée à 4kHz OD, puis remontée à 8kHz | Encoche 4kHz typique traumatisme sonore |
| `#1 [Baseline]` | 0.553 | OG quasi-plate à ~-10 dB, OD plate à ~30 dB | Asymétrie importante |
| `#8 [Periodic]` | 0.453 | Profil très irrégulier OG | Fluctuations / variabilité |
| `#7 [Periodic]` | 0.444 | Chute à 4kHz OD | Encoche 4kHz |
| `#2 [Baseline]` | 0.545 | Perte plate OG à 40+ dB | Perte conductive ou mixte |

---

### Figure 3 — UMAP Erreur de reconstruction

![UMAP reconstruction](results/figures/umap_reconstruction.png)

**UMAP** (Uniform Manifold Approximation and Projection) est une technique de réduction de dimension : elle prend les 15 features de chaque patient et les projette en **2D**, en essayant de préserver les distances relatives.

```
Chaque point = 1 patient
Position     = calculée à partir des 15 features (pas des axes réels)
Couleur      = erreur de reconstruction de l'autoencoder

  Vert foncé  → erreur faible → profil bien reconnu → NORMAL
  Jaune       → erreur intermédiaire
  Rouge       → erreur élevée → profil inhabituel   → ANORMAL
```

Le point **rouge isolé** en bas à droite (x≈3.9, y≈9.75) est le patient `#0`, celui avec la plus grande erreur (1.287). Sa position éloignée des autres points confirme qu'il est structurellement différent du reste du groupe.

---

### Figure 4 — UMAP Catégorie de visite

![UMAP visit category](results/figures/umap_visit_category.png)

Même projection UMAP, mais la couleur indique le **type de visite** :

```
Bleu foncé  (0) = Baseline  (visite initiale)
Bleu clair  (1) = Periodic  (visite de suivi)
```

Les deux couleurs sont mélangées → le type de visite **n'explique pas** les différences entre patients. Ce sont les profils audiométriques eux-mêmes qui créent la structure dans les données.

> Si les Baseline et Periodic formaient des clusters séparés, cela indiquerait une évolution systématique de l'audition entre les visites, ce qui n'est pas le cas ici.

---

## 6. Fichiers de sortie

Après exécution, le dossier `results/` contient :

```
results/
├── scores.csv                  ← scores de chaque patient par les 3 méthodes
├── isolation_forest.joblib     ← modèle Isolation Forest sérialisé
├── scaler.joblib               ← StandardScaler (pour de nouvelles données)
├── imputer.joblib              ← SimpleImputer (pour de nouvelles données)
├── autoencoder.pt              ← poids de l'Autoencoder PyTorch
└── figures/
    ├── score_distributions.png ← histogrammes des scores
    ├── top_anomalies.png       ← audiogrammes des 6 cas les plus anormaux
    ├── umap_reconstruction.png ← carte 2D colorée par score
    └── umap_visit_category.png ← carte 2D colorée par type de visite
```

### Structure de `scores.csv`

| Colonne | Méthode | Interprétation |
|---|---|---|
| `anomaly_score_if` | Isolation Forest | Score brut (plus négatif = plus anormal) |
| `anomaly_flag_if` | Isolation Forest | `1` = détecté anormal |
| `reconstruction_error` | Autoencoder | MSE (plus grand = plus anormal) |
| `anomaly_flag_ae` | Autoencoder | `1` = dans les 5% pires |
| `pca_reconstruction_error` | PCA | MSE PCA (plus grand = plus anormal) |
| `anomaly_flag_pca` | PCA | `1` = dans les 5% pires |
| `anomaly_consensus` | Tous | `1` si ≥ 2 méthodes d'accord |

---

## 7. Lancer le pipeline

### Commande de base

```bash
python main.py --data "JSON reports/"
```

### Toutes les options

```bash
python main.py \
  --data "JSON reports/"   # dossier JSON ou fichier unique
  --mode abs               # abs (valeurs brutes) ou delta (évolution vs Baseline)
  --contamination 0.05     # proportion attendue d'anomalies (0.05 = 5%)
  --epochs 100             # epochs d'entraînement de l'autoencoder
  --output-dir results/    # dossier de sortie
  --device cpu             # cpu ou cuda (GPU)
  --no-plots               # désactive la génération des figures
```

### Quand utiliser le mode `delta` ?

```
Mode abs   → compare les audiogrammes entre eux dans l'absolu
             Utile si les patients sont différents et qu'on cherche des profils rares

Mode delta → compare l'évolution de chaque patient par rapport à son propre Baseline
             Utile pour détecter une dégradation progressive même chez un patient
             déjà malentendant (sa Baseline est déjà mauvaise, mais s'empire-t-elle ?)

             ⚠ Nécessite au moins 1 Baseline + 1 Periodic/Depart par patient
```

### Appliquer le pipeline à de nouveaux patients

Les modèles sauvegardés (`scaler.joblib`, `imputer.joblib`, `autoencoder.pt`) peuvent être rechargés pour scorer de nouveaux audiogrammes sans ré-entraîner :

```python
import joblib, torch
from src.features import build_feature_matrix, preprocess

scaler  = joblib.load("results/scaler.joblib")
imputer = joblib.load("results/imputer.joblib")
if_model = joblib.load("results/isolation_forest.joblib")

feature_df, _ = build_feature_matrix(new_df)
X_new, _, _   = preprocess(feature_df, scaler=scaler, imputer=imputer, fit=False)
scores = if_model.score_samples(X_new)
```

---

## 8. Glossaire — Concepts expliqués simplement

> Cette section explique tous les termes techniques du pipeline, sans prérequis en audiologie ni en machine learning.

---

### Audiogramme

C'est le "graphique de l'audition" d'un patient. On lui fait entendre des sons purs à différentes fréquences (des graves aux aigus) et on note le volume minimal (en dB) à partir duquel il les entend.

```
Fréquence (Hz) : graves ──────────────────────────────▶ aigus
                  250    500   1k    2k    4k    8k

Volume (dB)
     0 dB  ─────────────────────────────────────────  ← entend tout (parfait)
    25 dB  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  ← seuil de normalité
    50 dB  ────────────────────────────────────────
   120 dB  ─────────────────────────────────────────  ← n'entend quasiment rien
```

Un point à 60 dB à 4000 Hz signifie : il faut un son de 60 dB (assez fort) pour que le patient entende les aigus autour de 4000 Hz. Plus le point est bas sur le graphique, plus la perte est importante.

---

### dB (décibel)

Unité de mesure du volume sonore. En audiologie, on parle de **seuil auditif** : le volume minimum pour qu'un patient entende un son.

```
  0 dB  → chuchotement imperceptible (audition parfaite)
 25 dB  → seuil de normalité
 60 dB  → niveau d'une conversation normale
 80 dB  → aspirateur
120 dB  → concert de rock / seuil de douleur
```

Un patient avec un seuil à 70 dB a besoin qu'on lui parle au niveau d'un aspirateur pour qu'il entende correctement.

---

### PTA — Pure Tone Average (Moyenne des Tons Purs)

C'est le résumé en un seul chiffre de la perte auditive d'une personne. On fait la **moyenne des seuils à 4 fréquences** qui couvrent la plage de la parole humaine :

```
PTA = ( seuil_500Hz + seuil_1000Hz + seuil_2000Hz + seuil_4000Hz ) / 4
```

Pourquoi ces 4 fréquences ? Parce qu'elles correspondent aux sons des voyelles et consonnes du langage courant. Si on entend mal dans cette plage, on a du mal à suivre une conversation.

```
PTA < 25 dB   → Audition normale     (pas de problème)
PTA 25–40 dB  → Perte légère         (difficultés au téléphone, dans le bruit)
PTA 40–70 dB  → Perte moyenne        (difficultés en conversation normale)
PTA > 70 dB   → Perte sévère/profonde (quasi aucune compréhension sans aide)
```

---

### STS — Standard Threshold Shift (Décalage Standard du Seuil)

C'est un **critère réglementaire** (norme OSHA aux États-Unis, repris en médecine du travail) pour détecter une dégradation auditive liée au travail.

**Principe :** on compare l'audiogramme actuel d'un travailleur à son audiogramme de référence (Baseline), uniquement aux fréquences 2000 Hz, 3000 Hz et 4000 Hz — la zone la plus sensible aux traumatismes sonores professionnels.

```
STS = moyenne( seuil_actuel(2k) + seuil_actuel(3k) + seuil_actuel(4k) )
    − moyenne( seuil_baseline(2k) + seuil_baseline(3k) + seuil_baseline(4k) )
```

**Seuil d'alerte :** si STS ≥ **10 dB**, c'est une alerte.

```
Exemple concret :
  Baseline (2017) :  25 dB / 30 dB / 35 dB  → moyenne = 30 dB
  Actuel   (2024) :  35 dB / 42 dB / 48 dB  → moyenne = 41.7 dB
  
  STS = 41.7 − 30 = 11.7 dB  → ⚠ STS positif, dégradation significative
```

En pratique : un STS ≥ 10 dB peut obligatoirement déclencher une action de l'employeur (révision du poste, port de protecteurs auditifs obligatoire...).

Dans ce pipeline, on calcule le STS séparément pour l'oreille gauche (`sts_L`) et droite (`sts_R`), et on génère un flag binaire (`has_sts_L/R = 1`) quand le seuil est dépassé.

---

### Encoche 4 kHz (Notch audiométrique)

C'est le signe radiographique du traumatisme sonore. Sur l'audiogramme, la perte est maximale autour de **4000 Hz**, puis remonte légèrement à 8000 Hz — formant une "encoche" (notch) caractéristique.

```
  Seuil (dB)
   0
  20  ·    ·
  40           ·         ← aggravation à 2k-4k Hz
  60               ·     ← pire à 4k Hz (zone lésée)
  40                  ·  ← remonte à 8k Hz  ← encoche typique
       250 500 1k 2k 4k 8k
```

Cause : les cellules ciliées de la cochlée (l'organe auditif interne) qui traitent les sons autour de 4000 Hz sont les plus exposées aux traumatismes sonores répétés (machines, casques, concerts...).

Dans le pipeline, la feature `high_freq_drop = seuil_8kHz − seuil_4kHz` capture ce pattern : si ce chiffre est négatif (le seuil remonte entre 4k et 8k), c'est le signe d'une encoche.

---

### PCA — Principal Component Analysis (Analyse en Composantes Principales)

**Analogie :** imaginez que vous avez 15 caractéristiques pour chaque patient (comme 15 dimensions). La PCA cherche les "axes" qui résument le mieux ces 15 dimensions en seulement 5 (ou moins).

```
Données originales (15 features)
        │
        ▼  PCA
5 composantes principales
(résumé de l'essentiel de la variance)
        │
        ▼  PCA inverse
15 features reconstruites (approximation)
        │
        ▼
Erreur = écart entre original et reconstruit
```

**Intuition :** si tous les patients ont des profils similaires, les 5 composantes suffisent à reconstruire leurs audiogrammes fidèlement. Un patient atypique ne sera pas bien résumé par ces 5 axes communs → son erreur de reconstruction sera élevée → il est suspect.

C'est la méthode la plus simple du pipeline. Elle sert de **référence (baseline)** pour évaluer si les méthodes plus complexes (Isolation Forest, Autoencoder) font vraiment mieux.

---

### Isolation Forest

**Analogie :** imaginez une forêt de questions aléatoires du type "ta feature X est-elle au-dessus ou en dessous de Y ?". Pour isoler un individu atypique, il faut peu de questions (il se distingue vite). Pour isoler un individu normal (qui ressemble aux autres), il en faut beaucoup.

```
Patient NORMAL   → beaucoup de questions nécessaires → difficile à isoler
Patient ANORMAL  → peu de questions nécessaires      → facile à isoler

Score :  -0.57  ← très négatif = anormal (peu de coupures nécessaires)
         -0.43  ← proche de 0 = normal (beaucoup de coupures)
```

L'algorithme construit 200 arbres de décision aléatoires et mesure la profondeur moyenne à laquelle chaque patient est isolé.

---

### Autoencoder (Réseau de neurones)

**Analogie :** c'est comme apprendre à résumer un livre en 6 mots, puis à le réécrire depuis ce résumé. Si le livre est "classique" (comme les autres), le résumé et la réécriture sont fidèles. Si le livre est totalement atypique, le résumé de 6 mots sera pauvre et la réécriture sera mauvaise.

```
15 features          6 valeurs (résumé)        15 features (réécriture)
    │                       │                          │
    └──── compression ──────┘────── décompression ─────┘
               ENCODEUR                 DÉCODEUR

Erreur = différence entre les 15 features originales et les 15 reconstruites
```

Le réseau s'entraîne sur tous les audiogrammes du dataset. Il apprend à bien reconstruire les profils fréquents. Un profil rare mal appris = erreur de reconstruction élevée = anomalie.

---

### UMAP — Uniform Manifold Approximation and Projection

**Analogie :** imaginez que vous avez 15 caractéristiques par patient → vous êtes dans un espace à 15 dimensions, impossible à visualiser. UMAP projette ces 15 dimensions sur une carte 2D (comme aplatir un globe terrestre sur une carte plate), en essayant de **conserver les distances** : deux patients proches dans les 15 dimensions restent proches sur la carte.

```
Chaque point sur la carte = 1 patient
Proximité sur la carte    = profils auditifs similaires
Point isolé               = profil radicalement différent des autres
```

Dans ce pipeline, la carte est colorée par l'erreur de reconstruction de l'Autoencoder : les points rouges (erreur élevée) sont les anomalies, les verts sont les normaux.

---

### Z-score (Standardisation)

Technique mathématique pour mettre toutes les features à la même échelle, afin qu'aucune ne "domine" les autres simplement par son ordre de grandeur.

```
nouvelle_valeur = (valeur − moyenne_de_la_colonne) / écart_type_de_la_colonne
```

Résultat : toutes les features ont une moyenne = 0 et un écart-type = 1.

```
Sans standardisation :
  seuil_8kHz = 80 dB    ← domine mathématiquement
  asymmetry  = 5 dB     ← ignoré car petit
  → le modèle croit que les aigus sont 16× plus importants

Avec standardisation :
  seuil_8kHz = +1.2     ← même importance
  asymmetry  = +0.8     ← même importance
  → le modèle traite toutes les features équitablement
```

---

### Imputation (médiane)

Quand une fréquence n'a pas été mesurée pour un patient (ex: 250 Hz absent), la case est vide (`NaN`). Les modèles ML ne savent pas gérer les cases vides.

On remplace chaque case vide par la **médiane** de cette colonne sur tout le dataset (la valeur "du milieu" si on trie toutes les valeurs de la colonne).

On choisit la médiane plutôt que la moyenne car elle est robuste aux valeurs extrêmes : un patient avec une perte sévère à 90 dB ne va pas trop fausser la valeur d'imputation.

---

### Consensus (≥ 2 modèles sur 3)

Pour éviter les fausses alarmes, on ne signale un patient comme anormal que si **au moins 2 des 3 méthodes** (Isolation Forest, Autoencoder, PCA) sont d'accord.

```
Isolation Forest : ANORMAL ✓
Autoencoder      : ANORMAL ✓
PCA              : normal  ✗

→ 2/3 d'accord → consensus = ANORMAL (alerte confirmée)
```

Chaque méthode a ses angles morts : l'une peut être sensible aux valeurs extrêmes, l'autre à la forme du profil. Le consensus rend le système plus robuste qu'un seul modèle.
