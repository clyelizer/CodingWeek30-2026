"""
src/train_model.py
==================
Pipeline d'entraînement des modèles ML — diagnostic pédiatrique de l'appendicite.

Paradigme fonctionnel strict :
  - Une fonction = une tâche précise et testable.
  - Pas d'état global mutable : chaque fonction reçoit ses entrées et retourne
    ses sorties explicitement.
  - Chaque fonction est couverte par exactement un test unitaire.

Modèles entraînés :
  - Logistic Regression  (baseline linéaire interprétable)
  - Random Forest        (modèle principal, robuste, supporte SHAP)
  - Gradient Boosting    (XGBoost-like, souvent meilleur sur données tabulaires)
  - SVM                  (comparaison)

Décision de conception :
  Chaque modèle est encapsulé dans un sklearn Pipeline avec StandardScaler.
  Cela évite le data leakage : le scaler est ajusté uniquement sur X_train,
  jamais sur X_test.

  Le déséquilibre de classes (40.6% positifs) est géré via class_weight='balanced'
  sur les modèles qui le supportent.
"""

from __future__ import annotations

import pathlib
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ---------------------------------------------------------------------------
# Définitions des modèles candidats
# ---------------------------------------------------------------------------


def build_logistic_regression() -> Pipeline:
    """
    Construit un pipeline Logistic Regression avec StandardScaler.

    Décision : LR est le baseline linéaire — rapide, interprétable, AUC fiable.
    class_weight='balanced' compense le déséquilibre 60/40.

    Assertion testée : le pipeline retourné contient bien 2 étapes.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    assert len(pipeline.steps) == 2
    return pipeline


def build_random_forest() -> Pipeline:
    """
    Construit un pipeline Random Forest avec StandardScaler.

    Décision : RF est le modèle principal du projet.
    - Robuste aux outliers
    - Supporte SHAP TreeExplainer (rapide)
    - class_weight='balanced' pour le déséquilibre

    Assertion testée : le pipeline retourné contient bien 2 étapes.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    assert len(pipeline.steps) == 2
    return pipeline


def build_gradient_boosting() -> Pipeline:
    """
    Construit un pipeline Gradient Boosting avec StandardScaler.

    Décision : GB souvent supérieur à RF sur données tabulaires médicales
    car il optimise directement la fonction de perte.

    Assertion testée : le pipeline retourné contient bien 2 étapes.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    random_state=42,
                ),
            ),
        ]
    )
    assert len(pipeline.steps) == 2
    return pipeline


def build_svm() -> Pipeline:
    """
    Construit un pipeline SVM (RBF kernel) avec StandardScaler.

    Décision : SVM inclus comme comparaison — performant sur petits datasets
    mais moins interprétable (pas de SHAP natif).
    probability=True requis pour predict_proba et AUC.

    Assertion testée : le pipeline retourné contient bien 2 étapes.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )
    assert len(pipeline.steps) == 2
    return pipeline


# ---------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------


def train_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Entraîne un pipeline sklearn sur les données d'entraînement.

    Retourne le pipeline fitted.

    Assertion testée : le pipeline fitted peut appeler predict sans lever d'erreur.
    """
    pipeline.fit(X_train, y_train)
    return pipeline


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    Calcule les métriques de performance sur le jeu de test.

    Métriques retournées :
      - roc_auc  : AUC-ROC (métrique principale — insensible au seuil)
      - f1       : F1-score (macro) — équilibre précision/rappel
      - accuracy : taux de bonne classification

    Décision : l'AUC-ROC est la métrique principale car le contexte médical
    exige de comparer les modèles indépendamment d'un seuil de décision.
    Le F1 macro est conservé pour tenir compte du déséquilibre de classes.

    Assertion testée : roc_auc ∈ [0, 1].
    """
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "f1": float(f1_score(y_test, y_pred, average="macro")),
        "accuracy": float((y_pred == y_test).mean()),
    }
    assert 0.0 <= metrics["roc_auc"] <= 1.0, f"AUC invalide : {metrics['roc_auc']}"
    return metrics


def select_best_model(
    results: dict[str, dict[str, float]],
) -> str:
    """
    Sélectionne le nom du meilleur modèle selon l'AUC-ROC.

    Paramètre
    ---------
    results : dict {nom_modèle: {roc_auc: float, ...}}

    Retourne
    --------
    str : nom du meilleur modèle.

    Assertion testée : le nom retourné est bien présent dans results.
    """
    best = max(results, key=lambda name: results[name]["roc_auc"])
    assert best in results
    return best


# ---------------------------------------------------------------------------
# Sauvegarde / chargement
# ---------------------------------------------------------------------------


def save_model(
    pipeline: Pipeline,
    name: str,
    output_dir: str | pathlib.Path,
) -> pathlib.Path:
    """
    Sauvegarde un pipeline entraîné dans models/<name>.joblib.

    Assertion testée : le fichier existe après sauvegarde.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}.joblib"
    joblib.dump(pipeline, out_path)
    assert out_path.exists(), f"Échec sauvegarde : {out_path}"
    return out_path


def load_model(path: str | pathlib.Path) -> Pipeline:
    """
    Charge un pipeline sklearn depuis un fichier .joblib.

    Assertion testée : l'objet chargé est bien un Pipeline sklearn.
    """
    model = joblib.load(path)
    assert isinstance(model, Pipeline), (
        f"Objet chargé n'est pas un Pipeline : {type(model)}"
    )
    return model


# ---------------------------------------------------------------------------
# Pipeline complet d'entraînement
# ---------------------------------------------------------------------------


def run_training(
    data_path: str | pathlib.Path,
    models_dir: str | pathlib.Path,
) -> dict[str, Any]:
    """
    Entraîne tous les modèles, évalue leurs performances, sauvegarde le meilleur.

    Retourne un dict avec :
      - "results"    : métriques de tous les modèles
      - "best_name"  : nom du meilleur modèle
      - "best_path"  : chemin vers le fichier .joblib du meilleur modèle
    """
    # Chargement des données
    data = joblib.load(data_path)
    X_train: pd.DataFrame = data["X_train"]
    X_test: pd.DataFrame = data["X_test"]
    y_train: pd.Series = data["y_train"]
    y_test: pd.Series = data["y_test"]

    # Catalogue des modèles à entraîner
    candidates = {
        "logistic_regression": build_logistic_regression(),
        "random_forest": build_random_forest(),
        "gradient_boosting": build_gradient_boosting(),
        "svm": build_svm(),
    }

    results: dict[str, dict[str, float]] = {}

    for name, pipeline in candidates.items():
        print(f"  Entraînement : {name} ...", end=" ", flush=True)
        fitted = train_model(pipeline, X_train, y_train)
        metrics = evaluate_model(fitted, X_test, y_test)
        results[name] = metrics
        print(f"AUC={metrics['roc_auc']:.4f}  F1={metrics['f1']:.4f}")
        # Sauvegarder tous les modèles (utile pour l'app)
        save_model(fitted, name, models_dir)

    best_name = select_best_model(results)
    best_path = pathlib.Path(models_dir) / f"{best_name}.joblib"

    print(
        f"\n  Meilleur modèle : {best_name}  (AUC={results[best_name]['roc_auc']:.4f})"
    )

    return {
        "results": results,
        "best_name": best_name,
        "best_path": best_path,
    }


# ---------------------------------------------------------------------------
# Exécution directe
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _ROOT = pathlib.Path(__file__).resolve().parent.parent
    data_path = _ROOT / "data" / "processed" / "processed_data.joblib"
    models_dir = _ROOT / "models"

    print("=== Entraînement des modèles ===")
    summary = run_training(data_path, models_dir)

    print("\n=== Résultats complets ===")
    for name, metrics in summary["results"].items():
        marker = " ← MEILLEUR" if name == summary["best_name"] else ""
        print(
            f"  {name:<25} AUC={metrics['roc_auc']:.4f}  F1={metrics['f1']:.4f}  Acc={metrics['accuracy']:.4f}{marker}"
        )
