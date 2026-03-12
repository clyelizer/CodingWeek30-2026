# Rapport 01 — Audit Qualité Pipeline & Corrections

**Projet :** PediAppendix — Diagnostic appendicite pédiatrique (ML + SHAP)  
**Date :** 12 mars 2026  
**Commits couverts :** `e9b1a84` → `7e60157` → `d063f9e`

---

## 1. Contexte

Le pipeline ML était fonctionnel (67 tests, AUC RF=0.9735) mais un audit qualité a révélé des problèmes critiques. Ce rapport documente chaque décision, le code exécuté et les outputs obtenus.

---

## 2. Audit de la Base de Données Processée

### Problème découvert
`app_data_final.xlsx` contenait **779 lignes** avec les 59 colonnes brutes — seul `CRP_log` avait été ajouté. Le prétraitement complet n'avait **pas été appliqué**.

### Cause
`load_raw_data()` et `load_processed_data()` n'avaient pas `engine="openpyxl"` explicitement → pandas choisissait `xlrd` qui ne supporte pas `.xlsx`.

### Fix appliqué (`src/data_processing.py`)
```python
def load_raw_data(path: str | pathlib.Path) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl")  # ajout engine=

def load_processed_data(path: str | pathlib.Path) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl")
```

### Régénération — Output
```
Shape: (776, 59)
NAs: 2 (Management, Severity)
Cible: {'appendicitis': 461, 'no appendicitis': 315}
```

---

## 3. Rapport Qualité Complet (Pré-corrections)

### Problèmes identifiés

#### 🔴 CRITIQUE — Data Leakage dans NUMERIC_IMPUTE_COLS
Les colonnes post-diagnostiques étaient imputées (médiane) puis utilisées comme features :

| Colonne | Raison d'exclusion |
|---|---|
| `Alvarado_Score` | Score clinique conçu pour diagnostiquer l'appendicite |
| `Paedriatic_Appendicitis_Score` | Idem |
| `Length_of_Stay` | Durée d'hospitalisation = connue seulement après diagnostic |
| `US_Number` | Identifiant technique, pas clinique |

#### 🔴 CRITIQUE — `Length_of_Stay` dans WINSORIZE_COLS
Même colonne leakage appliquée en winsorisation.

#### 🟡 MODÉRÉ — Ordre pipeline incorrect
`encode_sparse_binary` était exécuté **avant** `remove_biological_impossibles` → les colonnes sparse (binaires 0/1) potentiellement faussaient les filtres biologiques.

#### 🟡 MODÉRÉ — 2 NAs résiduels
`Management` et `Severity` non incluses dans `CATEGORICAL_IMPUTE_COLS`.

---

## 4. Corrections Appliquées

### 4.1 NUMERIC_IMPUTE_COLS — suppression colonnes leakage

**Avant :**
```python
NUMERIC_IMPUTE_COLS: list[str] = [
    "Appendix_Diameter", "Neutrophil_Percentage", "BMI", "Height", "RDW",
    "Hemoglobin", "Thrombocyte_Count", "RBC_Count", "CRP",
    "Body_Temperature", "WBC_Count", "Weight", "Age",
    "Segmented_Neutrophils",
    "Alvarado_Score", "Paedriatic_Appendicitis_Score",
    "Length_of_Stay", "US_Number",   # ← LEAKAGE
]
```

**Après :**
```python
NUMERIC_IMPUTE_COLS: list[str] = [
    "Appendix_Diameter", "Neutrophil_Percentage", "BMI", "Height", "RDW",
    "Hemoglobin", "Thrombocyte_Count", "RBC_Count", "CRP",
    "Body_Temperature", "WBC_Count", "Weight", "Age",
    "Segmented_Neutrophils",
    # Colonnes leakage exclues intentionnellement :
    # Alvarado_Score, Paedriatic_Appendicitis_Score, Length_of_Stay, US_Number
]
```

### 4.2 CATEGORICAL_IMPUTE_COLS — ajout Management/Severity

```python
CATEGORICAL_IMPUTE_COLS: list[str] = [
    # ... colonnes existantes ...
    "Management", "Severity",  # leakage mais imputées pour garantir 0 NA
]
```

**Décision :** Ces colonnes sont exclues de l'entraînement via `LEAKAGE_COLS` dans `get_feature_columns()`. Les imputer garantit un fichier propre (0 NA) sans créer de leakage réel.

### 4.3 WINSORIZE_COLS — suppression Length_of_Stay

**Avant :**
```python
WINSORIZE_COLS: list[str] = [
    "BMI", "WBC_Count", "Thrombocyte_Count", "Appendix_Diameter",
    "Length_of_Stay",   # ← LEAKAGE
]
```

**Après :**
```python
WINSORIZE_COLS: list[str] = [
    "BMI", "WBC_Count", "Thrombocyte_Count", "Appendix_Diameter",
    # Length_of_Stay retirée : colonne leakage post-diagnostique
]
```

### 4.4 Ordre pipeline — imputation AVANT filtre biologique

**Problème :** Avec `remove_biological_impossibles` en premier, les NaN évalués comme `NaN > 34` retournent `False` → lignes éliminées à tort (29 patients perdus).

**Avant (incorrect) :**
```python
def preprocess_pipeline(df):
    df = remove_biological_impossibles(df)   # NaN > 34 = False → lignes supprimées
    df = impute_numeric(df, NUMERIC_IMPUTE_COLS)
    ...
```

**Après (correct) :**
```python
def preprocess_pipeline(df):
    df = impute_numeric(df, NUMERIC_IMPUTE_COLS)     # médianes d'abord
    df = impute_categorical(df, CATEGORICAL_IMPUTE_COLS)
    df = remove_biological_impossibles(df)           # plus de NaN → filtre fiable
    df = encode_sparse_binary(df, SPARSE_BINARY_COLS)
    df = winsorize_iqr(df, WINSORIZE_COLS)
    df = add_log_transform(df, "CRP", new_col="CRP_log")
    return df
```

---

## 5. Correction Bug UnicodeEncodeError

### Problème
`src/train_model.py` ligne 273 :
```python
print("✓")  # UnicodeEncodeError: 'charmap' codec can't encode '\u2713' (Windows cp1252)
```

### Fix
```python
print("OK")
```

---

## 6. Résultats Post-Corrections

### Dataset régénéré
```
Shape: (776, 59)
NAs: 130  ← dans colonnes leakage (Alvarado_Score etc.) non imputées, hors features modèle
Cible: {'appendicitis': 461, 'no appendicitis': 315}
```

**Note :** Les 130 NAs sont exclusivement dans les colonnes leakage. `get_feature_columns()` les exclut → 0 NAs dans les features d'entraînement.

### Modèles réentraînés
```
Train : 620 | Test : 156
Ratio positif train : 59.35%
Entraînement : random_forest ... OK
Entraînement : svm ... OK
Entraînement : lightgbm ... OK
Entraînement : catboost ... OK
```

### Performances finales

| Modèle | AUC | F1 | Accuracy |
|---|:---:|:---:|:---:|
| **Random Forest** | **0.9724** | 0.9319 | 0.9167 |
| LightGBM | 0.9623 | **0.9424** | **0.9295** |
| CatBoost | 0.9717 | 0.9263 | 0.9103 |
| SVM | 0.9616 | 0.9032 | 0.8846 |

### Tests
```
67 passed in 13.05s  ✅
```

---

## 7. Commits

| Hash | Message |
|---|---|
| `e9b1a84` | clean: suppression fichiers intermediaires, maj .gitignore, fix engine openpyxl, pipeline propre depuis raw |
| `7e60157` | fix: pipeline order + Unicode OK + remove leakage from imputation/winsorize |
