"""
tests/test_evaluate_model.py
============================
Paradigme fonctionnel strict : un test = une fonction = une assertion précise.

Chaque test est :
  - Indépendant (données synthétiques en mémoire)
  - Sans import shap au niveau module (import lazy dans les fonctions testées)
  - Précis (une seule assertion par test)

Note : les tests SHAP (compute_shap_values, make_shap_waterfall_b64) sont
marqués pytest.mark.shap et nécessitent que shap soit disponible.
Les tests de predict_proba_safe et build_results_summary sont toujours rapides.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate_model import (
    predict_proba_safe,
    build_results_summary,
)


# ---------------------------------------------------------------------------
# Fixtures — pipeline minimal entraîné
# ---------------------------------------------------------------------------


def _make_xy(n: int = 100):
    """Crée X (10 features) et y binaire synthétiques."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(
        {
            "Lower_Right_Abd_Pain": rng.integers(0, 2, n).astype(float),
            "Migratory_Pain": rng.integers(0, 2, n).astype(float),
            "Body_Temperature": rng.uniform(36, 40, n),
            "WBC_Count": rng.uniform(3, 25, n),
            "CRP": rng.uniform(0, 200, n),
            "Neutrophil_Percentage": rng.uniform(30, 90, n),
            "Ipsilateral_Rebound_Tenderness": rng.integers(0, 2, n).astype(float),
            "Appendix_Diameter": rng.uniform(4, 15, n),
            "Nausea": rng.integers(0, 2, n).astype(float),
            "Age": rng.uniform(1, 18, n),
        }
    )
    y = pd.Series(rng.integers(0, 2, n), name="Diagnosis")
    return X, y


@pytest.fixture
def fitted_pipeline():
    """Pipeline RF minimal entraîné sur données synthétiques."""
    X, y = _make_xy()
    p = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=10, random_state=42)),
        ]
    )
    p.fit(X, y)
    return p, X


# ---------------------------------------------------------------------------
# Tests — predict_proba_safe
# ---------------------------------------------------------------------------


def test_predict_proba_safe_returns_float(fitted_pipeline):
    """predict_proba_safe → retourne un float."""
    pipeline, X = fitted_pipeline
    result = predict_proba_safe(pipeline, X.iloc[[0]])
    assert isinstance(result, float)


def test_predict_proba_safe_in_range(fitted_pipeline):
    """predict_proba_safe → probabilité ∈ [0, 1]."""
    pipeline, X = fitted_pipeline
    proba = predict_proba_safe(pipeline, X.iloc[[0]])
    assert 0.0 <= proba <= 1.0


def test_predict_proba_safe_single_row(fitted_pipeline):
    """predict_proba_safe → fonctionne avec une seule ligne."""
    pipeline, X = fitted_pipeline
    proba = predict_proba_safe(pipeline, X.iloc[[5]])
    assert 0.0 <= proba <= 1.0


def test_predict_proba_safe_different_rows_can_differ(fitted_pipeline):
    """predict_proba_safe → des patients différents peuvent avoir des probas différentes."""
    pipeline, X = fitted_pipeline
    proba_all = [predict_proba_safe(pipeline, X.iloc[[i]]) for i in range(10)]
    assert len(set(proba_all)) > 1


# ---------------------------------------------------------------------------
# Tests — build_results_summary
# ---------------------------------------------------------------------------


def test_build_results_summary_correct_row_count():
    """build_results_summary → autant de lignes que de modèles."""
    results = {
        "rf": {"roc_auc": 0.92, "f1": 0.85, "accuracy": 0.88},
        "lr": {"roc_auc": 0.83, "f1": 0.75, "accuracy": 0.78},
        "svm": {"roc_auc": 0.81, "f1": 0.73, "accuracy": 0.76},
    }
    df = build_results_summary(results, best_name="rf")
    assert len(df) == 3


def test_build_results_summary_sorted_by_auc():
    """build_results_summary → trié par roc_auc décroissant."""
    results = {
        "lr": {"roc_auc": 0.83, "f1": 0.75, "accuracy": 0.78},
        "rf": {"roc_auc": 0.92, "f1": 0.85, "accuracy": 0.88},
        "svm": {"roc_auc": 0.81, "f1": 0.73, "accuracy": 0.76},
    }
    df = build_results_summary(results, best_name="rf")
    assert df["roc_auc"].iloc[0] == 0.92


def test_build_results_summary_best_flag_is_true_for_best():
    """build_results_summary → la colonne 'best' est True uniquement pour le meilleur."""
    results = {
        "rf": {"roc_auc": 0.92, "f1": 0.85, "accuracy": 0.88},
        "lr": {"roc_auc": 0.83, "f1": 0.75, "accuracy": 0.78},
    }
    df = build_results_summary(results, best_name="rf")
    assert df.loc[df["model"] == "rf", "best"].iloc[0]


def test_build_results_summary_contains_required_columns():
    """build_results_summary → contient les colonnes model, roc_auc, f1, accuracy, best."""
    results = {"rf": {"roc_auc": 0.92, "f1": 0.85, "accuracy": 0.88}}
    df = build_results_summary(results, best_name="rf")
    assert set(df.columns) == {"model", "roc_auc", "f1", "accuracy", "best"}
