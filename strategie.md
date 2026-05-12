# Stratégie — Détection d'anomalies audiométriques

## Objectif

Détecter des anomalies audiométriques avec une **haute précision** : quand le modèle flag un cas, il doit avoir raison.

---

## Contexte

- **5 000 audiogrammes** issus de la base Odyo (baseline, périodique, départ)
- **Modèle non-supervisé** : Isolation Forest + Autoencoder + PCA → `anomaly_consensus`
- **Règles cliniques** : NIHL, Menière, STS → `nihl_flag`, `meniere_flag`, `sts_flag`

### Résultats actuels (post-entraînement non-supervisé)

| Groupe | % | Interprétation |
|--------|---|----------------|
| Accord (ML + règle) | 58.5% | Les deux systèmes sont d'accord |
| Règle seule | 39.2% | Règle détecte, ML rate → angle mort du ML |
| ML seul | 0.9% | ML détecte, règle rate → patterns nouveaux ou faux positifs |

**Constat clé** : le ML sous-détecte massivement par rapport à la règle. Sans labels, impossible de savoir qui a raison.

---

## Stratégie globale

### Phase 1 — Diagnostic (en cours)

130 audiogrammes envoyés à l'audiologiste pour labellisation, répartis en 5 groupes :

| Groupe | N | But |
|--------|---|-----|
| Règle seule | 35 | Comprendre l'angle mort du ML |
| ML seul | 20 | Vrais positifs ou bruit ? |
| Zone grise (P50–P90) | 40 | Calibrer la frontière de décision |
| Consensus positif | 35 | Valider les vrais positifs |
| Consensus négatif | 20 | Valider les vrais négatifs |

### Phase 2 — Quand les 130 labels arrivent

**Immédiatement protéger le test set :**

```
130 labels
├── 30 cas → test set fixe (jamais touchés jusqu'à l'évaluation finale)
└── 100 cas → diagnostic + entraînement
```

Les 30 cas du test set = ~6 par groupe, stratifiés sur catégorie de visite / genre.

**Diagnostic sur les 100 cas :**

Calculer précision / rappel / F1 pour chaque stratégie :
- Règle seule
- ML seul
- Consensus (règle ET ML) ← probablement la meilleure précision
- Classifieur supervisé entraîné sur les 100 labels

→ Notebook `04_evaluation.ipynb`

### Phase 3 — Active learning (2–3 tours)

Entraîner un **Random Forest** (ou XGBoost) sur les features audiométriques avec les 100 labels.  
Pourquoi pas du deep learning : trop peu de données, overfitting assuré.

À chaque tour :
1. Le modèle tourne sur les ~4 900 cas non-labellisés
2. Il identifie les 50 cas les plus **incertains** (probabilité proche de 0.5)
3. Ces 50 cas sont envoyés à l'audiologiste
4. On réentraîne avec les nouveaux labels

```
Tour 1 : 100 labels → modèle v1 → 50 cas incertains → audiologiste
Tour 2 : 150 labels → modèle v2 → 50 cas incertains → audiologiste
Tour 3 : 200 labels → modèle v3 → évaluation finale
```

Total estimé : ~230 labels pour un modèle robuste (vs 800–1000 en labellisant aléatoirement).

→ Notebook `05_active_learning.ipynb`

### Phase 4 — Évaluation finale

Sortir les 30 cas du tiroir pour la première fois.  
Évaluer le modèle v3 sur ce test set :

- **Précision** = parmi les cas flaggés, combien sont vrais positifs
- **Rappel** = parmi les vrais positifs, combien ont été flaggés
- **F1** = moyenne harmonique précision/rappel
- **Intervalles de confiance** par bootstrap (important : 30 cas = IC larges)
- **Comparaison baseline** : règle seule sur les mêmes 30 cas

Si le modèle v3 > règle seule en précision → le ML apporte de la valeur réelle.

---

## Vue d'ensemble

```
Maintenant
    │
    ▼
130 labels arrivent
    │
    ├── 30 → test set (tiroir, ne pas toucher)
    └── 100 → diagnostic + modèle v1
                │
                ▼
           50 cas incertains → audiologiste (tour 2)
                │
                ▼
           modèle v2 → 50 cas incertains → audiologiste (tour 3)
                │
                ▼
           modèle v3 → évaluation sur les 30 du tiroir
                │
                ▼
        Précision / F1 finale avec intervalles de confiance
```

---

## Notebooks à produire

| Notebook | Quand | Contenu |
|----------|-------|---------|
| `04_evaluation.ipynb` | Dès réception des 130 labels | Diagnostic, split train/test, métriques baseline |
| `05_active_learning.ipynb` | Après le diagnostic | Modèle supervisé, sélection des cas incertains, export CSV audiologiste |

---

## Décisions clés à garder en tête

- Le test set est **intouchable** jusqu'à l'évaluation finale — ne jamais s'en servir pour choisir des hyperparamètres
- L'objectif est la **précision**, pas le rappel — mieux vaut rater une anomalie que d'alarmer à tort
- Le consensus (règle ET ML) est probablement la meilleure stratégie à court terme sans labels
- Les intervalles de confiance sur 30 cas de test seront larges (~±0.10 sur F1) — à mentionner dans tout rapport
