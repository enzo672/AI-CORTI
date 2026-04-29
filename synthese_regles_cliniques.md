---
title: "Règles cliniques — Pipeline CORTI"
author: "Enzo Kara — Avril 2026"
lang: fr
geometry: "margin=2cm, top=2cm, bottom=2cm"
fontsize: 11pt
colorlinks: true
---

# Principe général

Chaque seuil est d'abord **corrigé par l'âge et le genre** (OSHA 29 CFR 1910.95 App. F) :

$$\text{Résidu} = \text{seuil mesuré} - \text{seuil normatif (âge, genre)}$$

Toutes les règles s'appliquent sur ces **résidus**, sauf mention contraire. Un audiogramme est marqué **anomalie** dès qu'une règle se déclenche.

---

# Règle 1 — Encoche NIHL (3, 4 ou 6 kHz)

Le NIHL peut s'exprimer à 3, 4 ou 6 kHz selon la fréquence d'exposition. La règle est appliquée aux trois fréquences, chacune comparée à ses voisins immédiats :

$$\boxed{\text{notch}(f) = T(f) - \frac{T(f_{\text{bas}}) + T(f_{\text{haut}})}{2}}$$

| Fréquence cible | Voisin bas | Voisin haut |
|:---:|:---:|:---:|
| 3 kHz | 2 kHz | 4 kHz |
| 4 kHz | 2 kHz | 8 kHz |
| 6 kHz | 4 kHz | 8 kHz |

Calculé sur résidus corrigés, pour chaque oreille séparément.

> **Déclenchement :** `notch(f) > 15 dB` sur au moins une fréquence, une oreille

*Référence : Coles et al. (2000), NIOSH 98-126*

---

# Règle 2 — Perte basses fréquences · Ménière

$$\boxed{\text{low\_freq\_pta} = \frac{T(250\,\text{Hz}) + T(500\,\text{Hz}) + T(1000\,\text{Hz})}{3}}$$

Calculé sur résidus corrigés (adaptation du critère Barany pour population multi-âge).

> **Déclenchement :** `low_freq_pta_OG > 25 dB` **OU** `low_freq_pta_OD > 25 dB`

*Référence : Barany Society (2015)*

---

# Règle 3 — Standard Threshold Shift · OSHA

Critère réglementaire de surveillance auditive en santé au travail. Nécessite un audiogramme de référence (Baseline) par salarié.

$$\boxed{\text{STS} = \frac{\Delta T(2\,\text{kHz}) + \Delta T(3\,\text{kHz}) + \Delta T(4\,\text{kHz})}{3}}$$

avec $\Delta T(f) = T_{\text{actuel}}(f) - T_{\text{baseline}}(f)$. La valeur à 3 kHz est interpolée si non mesurée directement.

> **Déclenchement :** `STS_OG >= 10 dB` **OU** `STS_OD >= 10 dB`

Applicable uniquement en mode longitudinal (visites Periodic / Depart vs Baseline).

*Référence : OSHA 29 CFR 1910.95(g)*

---

# Autres règles

| Règle | Critère | Référence |
|:---|:---|:---|
| Perte sévère | PTA moyen > 65 dB (seuils bruts) | WHO Grade 3–4 |
| Asymétrie inter-oreilles | Ecart > 40 dB sur au moins 1 fréquence | AAO-HNS 1994 |
| Perte unilatérale soudaine | Chute > 30 dB sur 3 fréquences consécutives | AAO-HNS 2019 |

---

# Prochaine étape

Sélectionner **50 audiogrammes** flaggés (top anomalies + cas borderline) pour relecture clinique, afin de calibrer les seuils sur données réelles.
