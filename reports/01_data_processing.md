# 01 — Pipeline de traitement des données

## Contexte

**Problème clinique :** Aide au diagnostic pédiatrique de l'appendicite.  
**Source :** `data/raw/data_finale.xlsx` — 776 patients, 27 variables.  
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
**Déséquilibre de classes :** 461 négatifs (59.4%) / 315 positifs (40.6%) — léger déséquilibre, géré par split stratifié et paramètre `class_weight` dans les modèles.

---

## Étapes du pipeline (`src/data_processing.py`)

### 1. `load_raw_data(path)`
Charge le fichier Excel brut. Assertion : DataFrame non vide.

### 2. `select_columns(df)`
Filtre le DataFrame pour ne conserver que les 10 features + la cible.  
**Décision :** suppression dès cette étape pour éviter de transporter des colonnes inutiles dans tout le pipeline.

### 3. `encode_binary_columns(df)`
Encode les 4 colonnes `yes/no` → `1/0` via un mapping explicite.  
**Décision :** mapping explicite (`{"yes": 1, "no": 0}`) plutôt que `LabelEncoder` pour garantir l'ordre indépendamment de l'ordre alphabétique.

### 4. `split_features_target(df)`
Sépare `X` (features) et `y` (cible). Retourne un tuple `(DataFrame, Series)`.

### 5. `split_train_test(X, y)`
Split stratifié 80/20 avec `random_state=42`.  
**Décision :** split stratifié obligatoire compte tenu du déséquilibre de classes — garantit que la proportion de positifs est préservée dans les deux ensembles.

### 6. `save_processed_data(...)`
Sauvegarde un fichier `data/processed/processed_data.joblib` contenant `X_train`, `X_test`, `y_train`, `y_test` et `feature_cols`.

---

## Décision architecturale : pas de scaling ici

Le scaling (StandardScaler) **n'est pas appliqué dans ce pipeline**.  
Il sera encapsulé dans un `sklearn.Pipeline` propre à chaque modèle lors de l'entraînement (`train_model.py`).

**Justification :** appliquer le scaling ici forcerait à scaler sur l'ensemble train+test, ce qui constituerait une **fuite de données (data leakage)** — le StandardScaler verraient les statistiques du test set lors du `fit`. En encapsulant le scaler dans le pipeline modèle, le `fit` est strictement limité aux données d'entraînement.

---

## Tests unitaires (`tests/test_data_processing.py`)

| Test | Fonction testée | Assertion |
|------|----------------|-----------|
| `test_load_raw_data_non_empty` | `load_raw_data` | `len(df) > 0` |
| `test_select_columns_keeps_only_expected_columns` | `select_columns` | `set(columns) == FEATURE_COLS + TARGET` |
| `test_select_columns_drops_extra_column` | `select_columns` | `"ColonneInutile" not in columns` |
| `test_encode_binary_columns_values_are_0_or_1` | `encode_binary_columns` | valeurs ∈ {0, 1} |
| `test_encode_binary_columns_yes_maps_to_1` | `encode_binary_columns` | `"yes"` → `1` |
| `test_split_features_target_same_length` | `split_features_target` | `len(X) == len(y)` |
| `test_split_features_target_X_has_correct_columns` | `split_features_target` | colonnes == `FEATURE_COLS` |
| `test_split_train_test_stratified_ratio` | `split_train_test` | `|ratio_train - ratio_test| < 0.05` |
| `test_split_train_test_sizes` | `split_train_test` | taille test ≈ 20% |
| `test_save_processed_data_file_exists` | `save_processed_data` | fichier `.joblib` créé |
| `test_run_pipeline_produces_output_file` | `run_pipeline` | fichier produit end-to-end |
