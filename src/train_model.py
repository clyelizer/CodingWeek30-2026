"""
src/train_model.py
Paradigme fonctionnel : une fonction = une tâche précise et testable.
Entraînement et sauvegarde des modèles ML.
"""

from __future__ import annotations

import pathlib
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from src.data_processing import (
    load_processed_data,
    optimize_memory,
    get_feature_columns,
    get_target_column,
    encode_target,
    LEAKAGE_COLS,
)

# ---------------------------------------------------------------------------
# 1. Préparation des données
# ---------------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Sépare X (features) et y (cible binaire encodée).
    Encode les colonnes catégorielles restantes en entiers.

    Returns
    -------
    (X, y) où X est numérique et y est int8 (0/1).
    """
    X = df[feature_cols].copy()
    y = encode_target(df[target_col])

    # Encoder les colonnes objet/category restantes
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        encoded = {col: LabelEncoder().fit_transform(X[col].astype(str)) for col in cat_cols}
        X = X.assign(**encoded)

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divise X et y en ensembles d'entraînement et de test.
    Stratification pour préserver l'équilibre des classes.

    Returns
    -------
    (X_train, X_test, y_train, y_test)
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# ---------------------------------------------------------------------------
# 2. Entraînement — un modèle par fonction
# ---------------------------------------------------------------------------

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    max_depth: int | None = None,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Entraîne un Random Forest avec class_weight='balanced'.

    Choix : robuste aux outliers, pas besoin de normalisation,
    supporte nativement l'importance des features (SHAP compatible).

    Returns
    -------
    Modèle entraîné.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    C: float = 1.0,
    kernel: str = "rbf",
    random_state: int = 42,
) -> SVC:
    """
    Entraîne un SVM avec probabilités activées (nécessaire pour ROC-AUC).

    Returns
    -------
    Modèle entraîné.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            C=C,
            kernel=kernel,
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        )),
    ])
    model.fit(X_train, y_train)
    return model


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> object:
    """
    Entraîne un LightGBM Classifier.

    Returns
    -------
    Modèle entraîné.
    """
    import lightgbm as lgb

    ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        scale_pos_weight=ratio,
        random_state=random_state,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    iterations: int = 300,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> object:
    """
    Entraîne un CatBoost Classifier.

    Returns
    -------
    Modèle entraîné.
    """
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        random_seed=random_state,
        verbose=0,
        auto_class_weights="Balanced",
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# 3. Sérialisation
# ---------------------------------------------------------------------------

def save_model(model: object, path: str | pathlib.Path) -> None:
    """
    Sauvegarde un modèle avec joblib.

    Parameters
    ----------
    model : modèle scikit-learn compatible
    path  : chemin de destination (.joblib)
    """
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | pathlib.Path) -> object:
    """
    Charge un modèle sauvegardé avec joblib.

    Returns
    -------
    Modèle désérialisé.
    """
    return joblib.load(path)


# ---------------------------------------------------------------------------
# 4. Pipeline d'entraînement complet
# ---------------------------------------------------------------------------

def run_training_pipeline(
    processed_data_path: str | pathlib.Path,
    model_dir: str | pathlib.Path = "models",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Exécute le pipeline complet :
      1. Chargement du dataset traité
      2. Optimisation mémoire
      3. Préparation features / cible
      4. Split train/test
      5. Entraînement des 4 modèles
      6. Sauvegarde

    Returns
    -------
    Dictionnaire {'model_name': model, 'X_test': ..., 'y_test': ...}
    """
    df = load_processed_data(processed_data_path)
    df = optimize_memory(df)

    feature_cols = get_feature_columns(df)
    target_col = get_target_column()

    X, y = prepare_features(df, feature_cols, target_col)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)

    print(f"Train : {len(X_train)} | Test : {len(X_test)}")
    print(f"Ratio positif train : {y_train.mean():.2%}")

    trainers = {
        "random_forest": train_random_forest,
        "svm":           train_svm,
        "lightgbm":      train_lightgbm,
        "catboost":      train_catboost,
    }

    models = {}
    for name, trainer in trainers.items():
        print(f"Entraînement : {name} ...", end=" ", flush=True)
        try:
            model = trainer(X_train, y_train, random_state=random_state)
            save_model(model, pathlib.Path(model_dir) / f"{name}.joblib")
            models[name] = model
            print("✓")
        except ImportError as e:
            print(f"⚠ ignoré ({e})")

    # Sauvegarder les données de test pour évaluation
    joblib.dump({"X_test": X_test, "y_test": y_test, "feature_cols": feature_cols},
                pathlib.Path(model_dir) / "test_data.joblib")

    return {**models, "X_test": X_test, "y_test": y_test}


if __name__ == "__main__":
    import sys

    data_path = pathlib.Path("data/processed/app_data_final.xlsx")
    if not data_path.exists():
        print(f"Fichier introuvable : {data_path}", file=sys.stderr)
        sys.exit(1)

    results = run_training_pipeline(data_path, model_dir="models")
    print("\nModèles sauvegardés dans models/")
