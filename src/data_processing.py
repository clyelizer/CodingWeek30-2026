"""
src/data_processing.py
Paradigme fonctionnel : une fonction = une tâche précise et testable.
Aucun effet de bord global — chaque fonction retourne un nouveau DataFrame.
"""

from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constantes — déclaration centrale, partagée avec les tests et le pipeline
# ---------------------------------------------------------------------------

NUMERIC_IMPUTE_COLS: list[str] = [
    "Appendix_Diameter", "Neutrophil_Percentage", "Alvarado_Score",
    "Paedriatic_Appendicitis_Score", "BMI", "Height", "RDW", "US_Number",
    "Hemoglobin", "Thrombocyte_Count", "RBC_Count", "CRP",
    "Body_Temperature", "WBC_Count", "Length_of_Stay", "Weight", "Age",
    "Segmented_Neutrophils",
]

CATEGORICAL_IMPUTE_COLS: list[str] = [
    "RBC_in_Urine", "Ketones_in_Urine", "WBC_in_Urine",
    "Ipsilateral_Rebound_Tenderness", "Sex", "Diagnosis", "Neutrophilia",
    "Migratory_Pain", "Lower_Right_Abd_Pain", "Contralateral_Rebound_Tenderness",
    "Coughing_Pain", "Nausea", "Loss_of_Appetite", "Dysuria", "Stool",
    "Peritonitis", "Psoas_Sign", "Appendix_on_US", "US_Performed",
    "Free_Fluids", "Diagnosis_Presumptive",
]

SPARSE_BINARY_COLS: list[str] = [
    "Abscess_Location", "Gynecological_Findings", "Conglomerate_of_Bowel_Loops",
    "Ileus", "Perfusion", "Enteritis", "Appendicolith", "Coprostasis",
    "Perforation", "Appendicular_Abscess", "Bowel_Wall_Thickening",
    "Lymph_Nodes_Location", "Target_Sign", "Meteorism",
    "Pathological_Lymph_Nodes", "Appendix_Wall_Layers",
    "Surrounding_Tissue_Reaction",
]

WINSORIZE_COLS: list[str] = [
    "BMI", "WBC_Count", "Length_of_Stay", "Thrombocyte_Count", "Appendix_Diameter",
]

# Colonnes exclues du modèle ML (post-diagnostiques / data leakage)
LEAKAGE_COLS: list[str] = [
    "Diagnosis_Presumptive", "Severity", "Management",
    "Length_of_Stay", "Alvarado_Score", "Paedriatic_Appendicitis_Score",
    "Perforation", "Appendicular_Abscess",
    "CRP",          # redondant avec CRP_log
    "US_Number",    # identifiant technique
]

TARGET_COL: str = "Diagnosis"


# ---------------------------------------------------------------------------
# 1. Chargement
# ---------------------------------------------------------------------------

def load_raw_data(path: str | pathlib.Path) -> pd.DataFrame:
    """Charge le fichier Excel brut et retourne un DataFrame non modifié."""
    return pd.read_excel(path, engine="openpyxl")


def load_processed_data(path: str | pathlib.Path) -> pd.DataFrame:
    """Charge le dataset traité (app_data_final.xlsx)."""
    return pd.read_excel(path, engine="openpyxl")


# ---------------------------------------------------------------------------
# 2. Imputation
# ---------------------------------------------------------------------------

def impute_numeric(
    df: pd.DataFrame,
    columns: list[str],
    strategy: str = "median",
) -> pd.DataFrame:
    """
    Impute les valeurs manquantes des colonnes numériques.

    Parameters
    ----------
    df : DataFrame source (non modifié)
    columns : colonnes à imputer
    strategy : 'median' (défaut) ou 'mean'

    Returns
    -------
    Nouveau DataFrame avec colonnes imputées.
    """
    fill_values = {}
    for col in columns:
        if col not in df.columns:
            continue
        fill_values[col] = df[col].median() if strategy == "median" else df[col].mean()
    return df.assign(**{col: df[col].fillna(val) for col, val in fill_values.items()})


def impute_categorical(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Impute les valeurs manquantes des colonnes catégorielles par la mode.

    Returns
    -------
    Nouveau DataFrame avec colonnes imputées.
    """
    modes = {}
    for col in columns:
        if col not in df.columns:
            continue
        mode_val = df[col].mode()
        if len(mode_val) > 0:
            modes[col] = mode_val[0]
    return df.assign(**{col: df[col].fillna(val) for col, val in modes.items()})


# ---------------------------------------------------------------------------
# 3. Encodage des colonnes éparses (≥75% NA) en présence/absence
# ---------------------------------------------------------------------------

def encode_sparse_binary(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Convertit les colonnes à très forte proportion de NA en indicateur binaire.
    1 = valeur présente dans le dataset original, 0 = manquant (examen non réalisé).

    Returns
    -------
    Nouveau DataFrame avec colonnes encodées en int8.
    """
    updates = {col: df[col].notna().astype(np.int8) for col in columns if col in df.columns}
    return df.assign(**updates)


# ---------------------------------------------------------------------------
# 4. Nettoyage des valeurs biologiquement impossibles
# ---------------------------------------------------------------------------

def remove_biological_impossibles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les lignes contenant des valeurs physiologiquement impossibles.

    Règles cliniques appliquées :
      - Body_Temperature ≤ 34 °C   : hypothermie incompatible avec le contexte
      - Hemoglobin > 20 g/dL        : impossible chez l'enfant
      - RDW > 30 %                  : artefact de mesure
      - RBC_Count > 8 T/L           : impossible chez l'enfant

    Returns
    -------
    Nouveau DataFrame filtré.
    """
    result = df.copy()
    if "Body_Temperature" in result.columns:
        result = result[result["Body_Temperature"] > 34]
    if "Hemoglobin" in result.columns:
        result = result[result["Hemoglobin"] <= 20]
    if "RDW" in result.columns:
        result = result[result["RDW"] <= 30]
    if "RBC_Count" in result.columns:
        result = result[result["RBC_Count"] <= 8]
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Winsorisation par IQR
# ---------------------------------------------------------------------------

def winsorize_iqr(
    df: pd.DataFrame,
    columns: list[str],
    factor: float = 1.5,
) -> pd.DataFrame:
    """
    Plafonne les valeurs extrêmes à [Q1 - factor·IQR, Q3 + factor·IQR].

    Parameters
    ----------
    columns : colonnes à winsoriser
    factor  : multiplicateur IQR (défaut 1.5)

    Returns
    -------
    Nouveau DataFrame avec colonnes winsoriées.
    """
    clips = {}
    for col in columns:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        clips[col] = df[col].clip(lower=Q1 - factor * IQR, upper=Q3 + factor * IQR)
    return df.assign(**clips)


# ---------------------------------------------------------------------------
# 6. Feature engineering
# ---------------------------------------------------------------------------

def add_log_transform(
    df: pd.DataFrame,
    column: str,
    new_col: str | None = None,
) -> pd.DataFrame:
    """
    Ajoute log1p(column) comme nouvelle colonne.

    Parameters
    ----------
    column  : colonne source (ex. 'CRP')
    new_col : nom de la colonne créée (défaut : column + '_log')

    Returns
    -------
    Nouveau DataFrame avec la colonne ajoutée.
    """
    target_name = new_col or f"{column}_log"
    return df.assign(**{target_name: np.log1p(df[column])})


# ---------------------------------------------------------------------------
# 7. Optimisation mémoire
# ---------------------------------------------------------------------------

def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Réduit l'empreinte mémoire par downcast systématique des types.

    Stratégie :
      - int64/int32 → int8 / int16 / int32 selon la plage de valeurs
      - float64     → float32
      - object (cardinalité < 50%) → category

    Returns
    -------
    Nouveau DataFrame avec types optimisés (copie).
    """
    result = df.copy()
    conversions: dict[str, type] = {}
    for col in result.columns:
        dtype = result[col].dtype
        if dtype in (np.dtype("int64"), np.dtype("int32")):
            c_min, c_max = result[col].min(), result[col].max()
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                conversions[col] = np.int8
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                conversions[col] = np.int16
            else:
                conversions[col] = np.int32
        elif dtype == np.dtype("float64"):
            conversions[col] = np.float32
        elif dtype == np.dtype("object"):
            if result[col].nunique() / max(len(result), 1) < 0.5:
                conversions[col] = "category"
    return result.astype(conversions)


# ---------------------------------------------------------------------------
# 8. Sélection des features (sans leakage)
# ---------------------------------------------------------------------------

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Retourne la liste des colonnes utilisables comme features ML.
    Exclut la cible, les colonnes post-diagnostiques et les identifiants.

    Returns
    -------
    Liste de noms de colonnes.
    """
    exclude = set(LEAKAGE_COLS) | {TARGET_COL}
    return [c for c in df.columns if c not in exclude]


def get_target_column() -> str:
    """Retourne le nom de la variable cible."""
    return TARGET_COL


def encode_target(series: pd.Series) -> pd.Series:
    """
    Encode la colonne cible en binaire : appendicitis=1, autre=0.

    Returns
    -------
    Série int8.
    """
    return (series.str.strip() == "appendicitis").astype(np.int8)


# ---------------------------------------------------------------------------
# 9. Pipeline complet (composition des fonctions ci-dessus)
# ---------------------------------------------------------------------------

def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique séquentiellement toutes les étapes de préprocessing.

    Ordre :
      1. Imputation numériques (médiane)
      2. Imputation catégorielles (mode)
      3. Encodage colonnes éparses (binaire)
      4. Suppression valeurs biologiquement impossibles
      5. Winsorisation IQR
      6. Log-transformation CRP

    Returns
    -------
    DataFrame prêt pour l'entraînement ML.
    """
    df = impute_numeric(df, NUMERIC_IMPUTE_COLS)
    df = impute_categorical(df, CATEGORICAL_IMPUTE_COLS)
    df = encode_sparse_binary(df, SPARSE_BINARY_COLS)
    df = remove_biological_impossibles(df)
    df = winsorize_iqr(df, WINSORIZE_COLS)
    df = add_log_transform(df, "CRP", new_col="CRP_log")
    return df
