"""
src/data_processing.py
======================
Pipeline de préparation des données — diagnostic pédiatrique de l'appendicite.

Paradigme fonctionnel : une étape = une fonction = une tâche précise.
 et testable.
  - Aucun effet de bord global : chaque fonction prend un DataFrame en entrée
    et retourne un nouveau DataFrame (pas de mutation en place).
  - Chaque fonction est couverte par exactement un test unitaire avec une
    assertion précise.

Étapes du pipeline :
  1. load_raw_data          → charge le fichier Excel (nettoyé ou brut)
  2. select_columns         → conserve uniquement les features + la cible
  3. encode_binary_columns  → encode yes/no → 1/0
  4. split_features_target  → sépare X et y
  5. split_train_test       → split stratifié 80/20
  6. save_processed_data    → exporte les fichiers .joblib dans data/processed/
"""

from __future__ import annotations

import pathlib
from typing import Tuple

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Constantes du projet
# ---------------------------------------------------------------------------

NUMERIC_FEATURES: list[str] = [
    "Body_Temperature",
    "WBC_Count",
    "CRP",
    "Neutrophil_Percentage",
    "Appendix_Diameter",
    "Age",
]

CATEGORICAL_FEATURES: list[str] = [
    "Lower_Right_Abd_Pain",
    "Migratory_Pain",
    "Ipsilateral_Rebound_Tenderness",
    "Nausea",
]

FEATURE_COLS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COL: str = "Diagnosis"

# Colonnes avec des valeurs textuelles "yes"/"no" à encoder en 1/0
BINARY_COLS: list[str] = [
    "Lower_Right_Abd_Pain",
    "Migratory_Pain",
    "Nausea",
    "Ipsilateral_Rebound_Tenderness",
]


# ---------------------------------------------------------------------------
# Fonctions du pipeline
# ---------------------------------------------------------------------------

def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimise la mémoire consommée par le DataFrame en convertissant 
    les types numériques vers les plus petits possibles (int8, float32, etc.).
    """
    df_opt = df.copy()
    start_mem = df_opt.memory_usage().sum() / 1024**2
    
    for col in df_opt.columns:
        col_type = df_opt[col].dtype
        
        # On ne traite que les colonnes purement numériques
        if pd.api.types.is_numeric_dtype(col_type):
            c_min = df_opt[col].min()
            c_max = df_opt[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df_opt[col] = df_opt[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df_opt[col] = df_opt[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df_opt[col] = df_opt[col].astype(np.int32)
                else:
                    df_opt[col] = df_opt[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df_opt[col] = df_opt[col].astype(np.float32)
                else:
                    df_opt[col] = df_opt[col].astype(np.float64)
                    
    end_mem = df_opt.memory_usage().sum() / 1024**2
    print(f"      Optimisation mémoire : {start_mem:.2f} MB \u2192 {end_mem:.2f} MB (Gain: {100*(start_mem-end_mem)/start_mem:.1f}%)")
    return df_opt



def load_raw_data(path: str | pathlib.Path) -> pd.DataFrame:
    """Charge le fichier Excel et retourne un DataFrame."""
    df = pd.read_excel(path)
    assert len(df) > 0, f"Le fichier {path} est vide."
    return df


def select_columns(
    df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Conserve uniquement les colonnes utiles (features + cible)."""
    cols = feature_cols + [target_col]
    missing = [c for c in cols if c not in df.columns]
    assert len(missing) == 0, f"Colonnes absentes de la DB : {missing}"

    df_selected = df[cols].dropna().reset_index(drop=True)
    return df_selected


def encode_binary_columns(
    df: pd.DataFrame,
    binary_cols: list[str] = BINARY_COLS,
) -> pd.DataFrame:
    """Encode les colonnes catégorielles binaires 'yes'/'no' en 1/0."""
    df_encoded = df.copy()
    mapping = {"yes": 1, "no": 0}
    for col in binary_cols:
        # Si c'est déjà numérique, on ne fait rien
        # Conversion en type object pour permettre l'assignation d'entiers si c'est un StringArray
        df_encoded[col] = df_encoded[col].astype(object).str.lower().map(mapping)
        unknown = df_encoded[col].isna().sum()
        assert unknown == 0, f"Colonne '{col}' contient des valeurs inconnues après encodage."
    return df_encoded


def split_features_target(
    df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    target_col: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Sépare le DataFrame en matrice de features X et vecteur cible y."""
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    assert len(X) == len(y), "X et y ont des longueurs différentes."
    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Effectue un split stratifié 80/20 train/test."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str | pathlib.Path,
) -> pathlib.Path:
    """Sauvegarde les données découpées dans un fichier .joblib unique."""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "processed_data.joblib"
    joblib.dump(
        {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_cols": list(X_train.columns),
        },
        out_path,
    )
    assert out_path.exists(), f"Échec de la sauvegarde : {out_path}"
    return out_path


from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path, target_col='Diagnosis', test_size=0.2, random_state=42):
    """Charge et prépare les données pour l'entraînement."""
    df_raw = load_raw_data(path)
    df_opt = optimize_memory(df_raw)
    df_sel = select_columns(df_opt, target_col=target_col)
    df_enc = encode_binary_columns(df_sel)
    
    X, y = split_features_target(df_enc, target_col=target_col)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=test_size, random_state=random_state)
    
    # Prétraitement : Normalisation
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    # Sauvegarde du préprocesseur pour usage ultérieur (App, Tests)
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/preprocessor.pkl')
    
    return X_train_scaled, X_test_scaled, y_train, y_test, list(X.columns)

def get_feature_names(preprocessor=None):
    """Retourne la liste des features de l'interface."""
    return FEATURE_COLS

def run_pipeline(
    raw_path: str | pathlib.Path,
    output_dir: str | pathlib.Path,
) -> pathlib.Path:
    """Enchaîne toutes les étapes du pipeline."""
    X_train, X_test, y_train, y_test, _ = load_and_preprocess(raw_path)
    out_path = save_processed_data(X_train, X_test, y_train, y_test, output_dir)
    return out_path


if __name__ == "__main__":
    _ROOT = pathlib.Path(__file__).resolve().parent.parent
    # On utilise le dataset nettoyé comme demandé
    source = _ROOT / "data" / "processed" / "data_finale.xlsx"
    out    = _ROOT / "data" / "processed"

    print(f"Lancement du pipeline sur {source}...")
    path = run_pipeline(source, out)
    print(f"Pipeline terminé \u2192 {path}")
