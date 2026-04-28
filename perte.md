# Types de pertes auditives — Anomalies audiométriques

---

## 1. Configurations de perte auditive

### Perte plate (Flat loss)
```
dB
 0  ─────────────────────────────
20
40  ●────────────────────────────●
60
    250  500  1k   2k   4k   8k
```
Toutes les fréquences sont touchées de façon uniforme.
Typique des pertes conductives (otosclérose, otite séreuse chronique) ou de certaines pertes neurosensorielles.

---

### Pente descendante / Ski-slope
```
dB
 0  ●──●
20       ●──●
40            ●
60                 ●
    250  500  1k   2k   4k   8k
```
Les aigus sont plus touchés que les graves. Configuration la plus fréquente.
Causes : presbyacousie, ototoxicité, séquelles de traumatisme sonore.

---

### Encoche 4 kHz (Notch audiométrique)
```
dB
 0  ●──●──●
20            ●
40                 ●        ←  pire à 4kHz
20                      ●   ←  remonte à 8kHz
    250  500  1k   2k   4k   8k
```
Pathognomonique d'une exposition au bruit (machines, casques, tirs).
Les cellules ciliées de la région 4 kHz sont les plus vulnérables.
L'encoche peut aussi apparaître à 3 kHz selon l'exposition.

---

### Perte ascendante (Rising)
```
dB
 0                      ●──●
20            ●──●
40  ●──●
    250  500  1k   2k   4k   8k
```
Les graves sont plus touchés que les aigus. Rare.
Évocateur de la maladie de Ménière (début d'évolution) ou d'une fistule périlymphatique.

---

### Courbe en U (Cookie-bite)
```
dB
 0  ●──●               ●──●
20
40       ●──●──●──●
    250  500  1k   2k   4k   8k
```
Les fréquences médiums sont plus touchées que les extrêmes.
Souvent d'origine génétique / congénitale.

---

### Courbe en U inversé (Tent / Mountain)
```
dB
 0       ●──●──●──●
20  ●                       ●
40
    250  500  1k   2k   4k   8k
```
Rare. Certaines causes ototoxiques ou variantes génétiques.

---

### Corner audiogram (Angle mort)
```
dB
 0
20
40
60
80  ●──●
100       ●──●──●──●──●──●
    250  500  1k   2k   4k   8k
```
Perte sévère à profonde sur toutes les fréquences sauf les graves.
Implantation cochléaire souvent discutée.

---

## 2. Types par origine

| Type | Siège de la lésion | Pattern typique |
|---|---|---|
| **Conductif** | Oreille externe ou moyenne (tympan, osselets) | Perte plane, max 60 dB, BC normale |
| **Neurosensoriel** | Cochlée ou nerf auditif | Variable, BC = CA, pas de RAC |
| **Mixte** | Conductif + neurosensoriel | BC dégradée + gap BC/CA |
| **Rétrocochléaire** | Nerf VIII / tronc cérébral | Asymétrie, mauvaise discrimination malgré seuils OK |

> **BC** = conduction osseuse · **CA** = conduction aérienne · **RAC** = réserve adaptative cochléaire

---

## 3. Anomalies spécifiques

### Asymétrie OG / OD
Différence > 15–20 dB sur plusieurs fréquences = drapeau rouge clinique.
Peut indiquer un neurinome acoustique (schwannome vestibulaire), une pathologie unilatérale, ou une simulation.

---

### Encoche de Carhart
Perte artificielle en conduction osseuse autour de 2 kHz, sans atteinte cochléaire réelle.
Signe quasi-pathognomonique de l'otosclérose. Disparaît après stapédectomie.

---

### Courbe d'ombre (Shadow curve)
Quand une oreille est profondément sourde, les sons très forts passent à travers le crâne vers la bonne oreille.
On mesure alors l'oreille saine en croyant mesurer la mauvaise — faux résultats sans masquage adéquat.

---

### Surdité brusque (SSNHL)
Perte neurosensorielle unilatérale apparaissant en < 72h, ≥ 30 dB sur 3 fréquences consécutives.
Urgence médicale. Pattern variable (plat, aigu, total).

---

### Fluctuante (Ménière)
La perte varie d'un examen à l'autre, surtout dans les graves.
Associée à acouphènes, plénitude auriculaire, vertiges.

---

### Ototoxicité
Médicaments (cisplatine, aminosides, hautes doses d'aspirine...) détruisent les cellules ciliées de la base de la cochlée (aigus en premier).
Pente descendante rapide, bilatérale, symétrique.

---

### Presbyacousie
Pente descendante bilatérale, symétrique, progressive avec l'âge.
Débute aux aigus (> 4 kHz), descend vers les fréquences de la parole.

---

## 4. Anomalies de fiabilité / non organiques

| Situation | Signe d'alerte |
|---|---|
| **Simulation / exagération** | Seuils très variables, incohérence BC/CA, test de Stenger positif |
| **Variabilité test-retest** | Différence > 10 dB sur la même fréquence lors de deux mesures rapprochées |
| **Fatigue auditive** | Seuils qui se dégradent en cours de test (SISI, TDT) |

---

## 5. Correspondance avec les features du pipeline

| Anomalie | Feature(s) concernée(s) |
|---|---|
| Pente descendante / presbyacousie | `high_freq_drop_L/R` élevé |
| Encoche 4 kHz | `high_freq_drop_L/R` négatif (remontée à 8kHz) |
| Asymétrie OG/OD | `asymmetry_mean` > 15–20 dB |
| Perte sévère globale | `PTA_L / PTA_R` > 70 dB |
| STS (dégradation professionnelle) | `sts_L / sts_R` ≥ 10 dB, `has_sts_L/R = 1` |
| Perte plate | `PTA` élevé sans `high_freq_drop` marqué |
