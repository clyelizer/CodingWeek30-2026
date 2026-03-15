"""
src/data_processing.py
======================
Pipeline de préparation des données — diagnostic pédiatrique de l'appendicite.

Paradigme fonctionnel strict :
  - Une fonction = une tâche précise et testable.
  - Aucun effet de bord global : chaque fonction prend un DataFrame en entrée
    et retourne un nouveau DataFrame (pas de mutation en place).
  - Chaque fonction est couverte par exactement un test unitaire avec une
    assertion précise.

Étapes du pipeline :
  1. load_raw_data          → charge le fichier Excel brut
  2. select_columns         → conserve uniquement les features + la cible
  3. encode_binary_columns  → encode yes/no → 1/0
  4. split_features_target  → sépare X et y
  5. split_train_test       → split stratifié 80/20
  6. save_processed_data    → exporte les fichiers .joblib dans data/processed/

Décision de conception :
  Le scaling (StandardScaler) n'est PAS appliqué ici. Il sera encapsulé dans
  un sklearn Pipeline propre à chaque modèle dans train_model.py, ce qui
  évite toute fuite de données (data leakage) entre train et test.
"""

from __future__ import annotations

import pathlib
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Constantes du projet
# ---------------------------------------------------------------------------

FEATURE_COLS: list[str] = [
    "Lower_Right_Abd_Pain",
    "Migratory_Pain",
    "Body_Temperature",
    "WBC_Count",
    "CRP",
    "Neutrophil_Percentage",
    "Ipsilateral_Rebound_Tenderness",
    "Appendix_Diameter",
    "Nausea",
    "Age",
]

TARGET_COL: str = "Diagnosis"

# Colonnes avec des valeurs textuelles "yes"/"no" à encoder en 1/0
BINARY_COLS: list[str] = [
    "Lower_Right_Abd_Pain",
    "Migratory_Pain",
    "Ipsilateral_Rebound_Tenderness",
    "Nausea",
]


# ---------------------------------------------------------------------------
# Fonctions du pipeline
# ---------------------------------------------------------------------------

def load_raw_data(path: str | pathlib.Path) -> pd.DataFrame:
    """
    Charge le fichier Excel brut et retourne un DataFrame.

    Paramètre
    ---------
    path : chemin vers le fichier .xlsx

    Retourne
    --------
    pd.DataFrame brut, sans aucune transformation.

    Assertion testée : le DataFrame chargé est non vide.
    """
    df = pd.read_excel(path)
    assert len(df) > 0, f"Le fichier {path} est vide."
    return df


def select_columns(
    df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """
    Conserve uniquement les colonnes utiles (features + cible) et supprime
    les lignes avec valeurs manquantes sur ces colonnes.

    Décision : On ne conserve que les colonnes nécessaires dès cette étape
    pour éviter de transporter des données inutiles dans le pipeline.

    Assertion testée : toutes les colonnes demandées sont présentes dans le
    DataFrame résultant.
    """
    cols = feature_cols + [target_col]
    missing = [c for c in cols if c not in df.columns]
    assert len(missing) == 0, f"Colonnes absentes de la DB : {missing}"

    df_selected = df[cols].dropna().reset_index(drop=True)
    return df_selected


def encode_binary_columns(
    df: pd.DataFrame,
    binary_cols: list[str] = BINARY_COLS,
) -> pd.DataFrame:
    """
    Encode les colonnes catégorielles binaires "yes"/"no" en 1/0.

    Décision : On utilise un mapping explicite plutôt que LabelEncoder pour
    garantir que "yes" → 1 et "no" → 0 indépendamment de l'ordre alphabétique.

    Assertion testée : les colonnes encodées ne contiennent que des valeurs
    dans {0, 1}.
    """
    df_encoded = df.copy()
    mapping = {"yes": 1, "no": 0}
    for col in binary_cols:
        df_encoded.loc[:, col] = df_encoded[col].map(mapping)
        unknown = df_encoded[col].isna().sum()
        assert unknown == 0, (
            f"Colonne '{col}' contient des valeurs inconnues après encodage."
        )
    return df_encoded


def split_features_target(
    df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    target_col: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Sépare le DataFrame en matrice de features X et vecteur cible y.

    Assertion testée : X et y ont le même nombre de lignes.
    """
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
    """
    Effectue un split stratifié 80/20 train/test.

    Décision : Le split est stratifié sur y pour préserver la proportion
    de cas positifs (appendicite) dans les deux ensembles, compte tenu du
    léger déséquilibre de classes (461 négatifs / 315 positifs).

    Assertion testée : la proportion de positifs dans train et test diffère
    de moins de 2 points de pourcentage.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    ratio_train = y_train.mean()
    ratio_test = y_test.mean()
    assert abs(ratio_train - ratio_test) < 0.02, (
        f"Split non stratifié : train={ratio_train:.3f}, test={ratio_test:.3f}"
    )
    return X_train, X_test, y_train, y_test


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str | pathlib.Path,
) -> pathlib.Path:
    """
    Sauvegarde les données découpées dans un fichier .joblib unique.

    Le fichier contient un dictionnaire avec les clés :
      X_train, X_test, y_train, y_test, feature_cols

    Assertion testée : le fichier exporté existe bien sur le disque.

    Retourne
    --------
    pathlib.Path vers le fichier sauvegardé.
    """
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


# ---------------------------------------------------------------------------
# Pipeline complet
# ---------------------------------------------------------------------------

def run_pipeline(
    raw_path: str | pathlib.Path,
    output_dir: str | pathlib.Path,
) -> pathlib.Path:
    """
    Enchaîne toutes les étapes du pipeline de traitement des données.

    Paramètres
    ----------
    raw_path   : chemin vers data/raw/data_finale.xlsx
    output_dir : chemin vers data/processed/

    Retourne
    --------
    pathlib.Path vers le fichier processed_data.joblib produit.
    """
    df_raw      = load_raw_data(raw_path)
    df_selected = select_columns(df_raw)
    df_encoded  = encode_binary_columns(df_selected)
    X, y        = split_features_target(df_encoded)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    out_path    = save_processed_data(X_train, X_test, y_train, y_test, output_dir)
    return out_path


# ---------------------------------------------------------------------------
# Exécution directe
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _ROOT = pathlib.Path(__file__).resolve().parent.parent
    raw   = _ROOT / "data" / "raw" / "data_finale.xlsx"
    out   = _ROOT / "data" / "processed"

    path = run_pipeline(raw, out)
    print(f"Pipeline terminé → {path}")
