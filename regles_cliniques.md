# Règles cliniques implémentées dans le pipeline CORTI

Document de validation — à relire avec le maître de stage

---

## Contexte

Le pipeline combine deux mécanismes de détection complémentaires :

1. **Modèles ML non supervisés** (Isolation Forest + Autoencoder + PCA) — détectent les audiogrammes dont la forme globale est inhabituelle par rapport à la population
2. **Règles déterministes cliniques** — détectent des patterns spécifiques définis dans la littérature audiologique

Un audiogramme est flaggé `anomaly_final` si au moins un des deux mécanismes le signale.

---

## Correction normative appliquée en amont

Avant tout calcul, les seuils mesurés sont corrigés par l'âge et le genre selon la norme **OSHA 29 CFR 1910.95 Appendix F** (Tables F-1 hommes, F-2 femmes), couvrant 1000–6000 Hz de 20 à 60 ans.

> **Résidu = seuil mesuré − seuil attendu pour cet âge/genre**
>
> Résidu ≈ 0 → audition normale pour cet âge  
> Résidu > 0 → perte au-delà de la norme → suspect  

Toutes les règles ci-dessous s'appliquent sur ces **résidus corrigés**, sauf mention contraire.

**Question pour le maître de stage :** La table OSHA Appendix F est-elle la référence utilisée en pratique en France, ou faut-il privilégier directement l'ISO 7029:2017 ?

---

## Règle 1 — Encoche 4 kHz (NIHL)

### Formule
```
notch_4k = T(4 kHz) − moyenne(T(2 kHz), T(8 kHz))
```
Calculé sur résidus corrigés, pour chaque oreille séparément.

Flag déclenché si : `notch_4k_OG > 15 dB` OU `notch_4k_OD > 15 dB`

### Interprétation
Une valeur positive signifie que le seuil à 4 kHz est spécifiquement élevé par rapport à ses fréquences voisines — patron caractéristique de la perte auditive induite par le bruit (NIHL).

Contrairement au simple calcul de chute haute fréquence (8 kHz − 4 kHz), cette formule capture les **deux côtés** de l'encoche et n'est pas biaisée par une perte haute fréquence générale (presbyacousie).

### Références
- Coles R.R.A. et al. (2000). *Guidelines on the diagnosis of noise-induced hearing loss for medicolegal purposes*. Clinical Otolaryngology, 25(4), 264–273.
- NIOSH (1998). *Criteria for a Recommended Standard: Occupational Noise Exposure*. Publication No. 98-126.

### Seuil : 15 dB
Valeur médiane dans la littérature. Certains auteurs utilisent 10 dB (plus sensible) ou 20 dB (plus spécifique).

**Question pour le maître de stage :** Le seuil de 15 dB est-il adapté au contexte CORTI ? Faut-il distinguer encoche légère (15–25 dB) et franche (> 25 dB) ?

---

## Règle 2 — Perte en basses fréquences (Ménière)

### Formule
```
low_freq_pta_corrigé = moyenne(résidu_250Hz, résidu_500Hz, résidu_1000Hz)
```
Calculé sur résidus corrigés par l'âge/genre.

Flag déclenché si : `low_freq_pta_OG > 25 dB` OU `low_freq_pta_OD > 25 dB`

### Interprétation
Mesure si les basses fréquences sont anormalement élevées **au-delà de ce qu'explique l'âge**. Un patient de 60 ans avec une presbyacousie naturelle ne sera pas flaggé ; un patient dont les BF dépassent significativement la norme de son âge le sera.

### Références
- Barany Society (2015). *Diagnostic criteria for Ménière's disease*. Journal of Vestibular Research, 25(1), 1–7.  
  Critère audiométrique : seuils absolus moy(250, 500, 1000 Hz) ≥ 25 dB HL sur l'oreille atteinte.

### Adaptation par rapport à Barany
Le critère original Barany utilise des **seuils absolus** (non corrigés par l'âge), calibré pour des patients jeunes. Sur une population multi-âge comme CORTI, cela génère trop de faux positifs (presbyacousie naturelle). La règle a donc été appliquée sur les **résidus corrigés** avec le même seuil de 25 dB.

Sur les 500 audiogrammes réels : 26 cas flaggés (5.2%), cohérent avec le taux d'anomalie attendu.

**Question pour le maître de stage :**
- L'adaptation sur résidus corrigés vous semble-t-elle cliniquement défendable ?
- Le seuil de 25 dB de résidu est-il pertinent ou faut-il l'ajuster ?
- Dans le contexte CORTI (santé au travail), la maladie de Ménière est-elle une pathologie à surveiller ou hors périmètre ?

---

## Résumé des features ajoutés au modèle ML

En plus des règles déterministes, deux features ont été ajoutés à la matrice d'apprentissage pour permettre au modèle ML de détecter ces patterns de façon non supervisée :

| Feature | Calcul | Interprétation |
|---|---|---|
| `notch_4k_L` / `notch_4k_R` | T(4kHz) − moy(T(2kHz), T(8kHz)) — corrigé | Profondeur encoche NIHL |
| `low_freq_pta_L` / `low_freq_pta_R` | moy(T(250), T(500), T(1000)) — **brut absolu** | Niveau BF absolu (Barany) |

La matrice passe de 21 à **25 features** avec ces ajouts.

---

## Ce qui n'a pas été implémenté (et pourquoi)

| Pattern | Raison |
|---|---|
| Cookie bite (perte en U 1-2 kHz) | Pas de critère audiométrique standardisé dans la littérature |
| Pente raide haute fréquence | Pattern décrit cliniquement mais pas de seuil consensuel publié |

**Question pour le maître de stage :** Ces patterns sont-ils présents dans la base CORTI ? Si oui, existe-t-il des critères utilisés en pratique ?

---

## Résultats de validation synthétique

Validation réalisée sur 120 audiogrammes synthétiques générés selon les critères cliniques ci-dessus (indépendants des règles de détection) + 40 audiogrammes normaux.

| Type | Référence génération | Rappel |
|---|---|---|
| Encoche 4kHz (NIHL) | Coles 2000 | 100% |
| Perte sévère (> 65 dB) | WHO Grade 3-4 | 100% |
| Asymétrie (> 40 dB) | AAO-HNS 1994 | 100% |
| Ménière (BF corrigé > 25 dB) | Barany 2015 adapté | 85% |
| Perte unilatérale soudaine | AAO-HNS 2019 | 100% |
| Pente raide HF | Littérature (non standardisé) | 100% |
| **Faux positifs sur normaux** | — | **0%** |

**Detection Score global : 97 / 100**

---

## Prochaine étape recommandée

Validation sur un échantillon réel : sélection de **50 audiogrammes** parmi les cas flaggés par le pipeline (top anomalies, cas borderline, faux négatifs potentiels) pour relecture par le maître de stage.

Cela permettrait d'obtenir une vérité terrain réelle et de calibrer les seuils des règles ci-dessus sur des données cliniques authentiques.
