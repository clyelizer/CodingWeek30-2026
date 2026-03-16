# 01 — Pipeline de traitement des données

## Contexte

**Problème clinique :** Aide au diagnostic pédiatrique de l'appendicite.  
**Source :** `data/raw/dataset.xlsx` — 776 patients, 27 variables.  
**Cible :** `Diagnosis` (0 = pas d'appendicite, 1 = appendicite) — déjà encodée binaire.

---

## Features retenues pour l'interface

| # | Colonne | Type | Domaine clinique |
|---|---------|------|-----------------|
| 1 | `Lower_Right_Abd_Pain` | Binaire (yes/no) | Examen clinique |
| 2 | `Migratory_Pain` | Binaire (yes/no) | Examen clinique |
| 3 | `Body_Temperature` | Numérique (°C) | Examen clinique |
| 4 | `WBC_Count` | Numérique (G/L) | Biologie |
| 5 | `CRP` | Numérique (mg/L) | Biologie |
| 6 | `Neutrophil_Percentage` | Numérique (%) | Biologie |
| 7 | `Ipsilateral_Rebound_Tenderness` | Binaire (yes/no) | Examen clinique |
| 8 | `Appendix_Diameter` | Numérique (mm) | Échographie |
| 9 | `Nausea` | Binaire (yes/no) | Examen clinique |
| 10 | `Age` | Numérique (années) | Démographique |

**Données manquantes :** aucune sur ces colonnes.  
**Déséquilibre de classes :** ~60% négatifs / ~40% positifs.

---

## Étapes du pipeline (`src/data_processing.py`)

### 1. `load_raw_data(path)`
Charge le fichier Excel brut. Assertion : DataFrame non vide.

### 2. `select_columns(df)`
Filtre le DataFrame pour ne conserver que les 10 features + la cible.

### 3. `encode_binary_columns(df)`
Encode les colonnes `yes/no` → `1/0`.

### 4. `split_features_target(df)`
Sépare `X` (features) et `y` (cible).

### 5. `split_train_test(X, y)`
Split stratifié 80/20.

### 6. `save_processed_data(...)`
Sauvegarde un fichier `data/processed/processed_data.joblib` contenant les splits.

---

## Décision architecturale : pas de scaling ici

Le scaling est appliqué via un préprocesseur lors de l'entraînement pour éviter le **data leakage**.

---

## Tests unitaires (`tests/test_data_processing.py`)

Les 11 tests unitaires valident chaque étape du pipeline, de la charge au split stratifié.
