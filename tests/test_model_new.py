"""
tests/test_model.py
===================
Paradigme fonctionnel strict : un test = une fonction = une assertion précise.

Chaque test est :
  - Indépendant (fixture scope=function, données synthétiques en mémoire)
  - Rapide (n_estimators réduit à 10 pour les tests)
  - Précis (une seule assertion par test)
"""

import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.train_model import (
    build_logistic_regression,
    build_random_forest,
    build_gradient_boosting,
    build_svm,
    train_model,
    evaluate_model,
    select_best_model,
    save_model,
    load_model,
)


# ---------------------------------------------------------------------------
# Fixtures — données synthétiques minimales
# ---------------------------------------------------------------------------

def _make_xy(n: int = 120):
    """Crée X (10 features) et y binaire synthétiques."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "Lower_Right_Abd_Pain":           rng.integers(0, 2, n).astype(float),
        "Migratory_Pain":                 rng.integers(0, 2, n).astype(float),
        "Body_Temperature":               rng.uniform(36, 40, n),
        "WBC_Count":                      rng.uniform(3, 25, n),
        "CRP":                            rng.uniform(0, 200, n),
        "Neutrophil_Percentage":          rng.uniform(30, 90, n),
        "Ipsilateral_Rebound_Tenderness": rng.integers(0, 2, n).astype(float),
        "Appendix_Diameter":              rng.uniform(4, 15, n),
        "Nausea":                         rng.integers(0, 2, n).astype(float),
        "Age":                            rng.uniform(1, 18, n),
    })
    y = pd.Series(rng.integers(0, 2, n), name="Diagnosis")
    return X, y


@pytest.fixture
def xy():
    return _make_xy()


@pytest.fixture
def trained_rf(xy):
    X, y = xy
    pipeline = build_random_forest()
    pipeline.named_steps["clf"].n_estimators = 10
    return train_model(pipeline, X, y)


# ---------------------------------------------------------------------------
# Tests — build_* : structure du pipeline
# ---------------------------------------------------------------------------

def test_build_logistic_regression_has_two_steps():
    """build_logistic_regression → pipeline a exactement 2 étapes."""
    p = build_logistic_regression()
    assert len(p.steps) == 2


def test_build_random_forest_has_two_steps():
    """build_random_forest → pipeline a exactement 2 étapes."""
    p = build_random_forest()
    assert len(p.steps) == 2


def test_build_gradient_boosting_has_two_steps():
    """build_gradient_boosting → pipeline a exactement 2 étapes."""
    p = build_gradient_boosting()
    assert len(p.steps) == 2


def test_build_svm_has_two_steps():
    """build_svm → pipeline a exactement 2 étapes."""
    p = build_svm()
    assert len(p.steps) == 2


def test_build_random_forest_last_step_is_clf():
    """build_random_forest → la dernière étape s'appelle 'clf'."""
    p = build_random_forest()
    assert p.steps[-1][0] == "clf"


# ---------------------------------------------------------------------------
# Tests — train_model
# ---------------------------------------------------------------------------

def test_train_model_returns_pipeline(xy):
    """train_model → retourne un objet Pipeline sklearn."""
    X, y = xy
    p = build_logistic_regression()
    fitted = train_model(p, X, y)
    assert isinstance(fitted, Pipeline)


def test_train_model_fitted_can_predict(xy):
    """train_model → le pipeline fitted peut appeler predict sans erreur."""
    X, y = xy
    p = build_logistic_regression()
    fitted = train_model(p, X, y)
    preds = fitted.predict(X)
    assert len(preds) == len(y)


# ---------------------------------------------------------------------------
# Tests — evaluate_model
# ---------------------------------------------------------------------------

def test_evaluate_model_roc_auc_in_range(trained_rf, xy):
    """evaluate_model → roc_auc ∈ [0, 1]."""
    X, y = xy
    metrics = evaluate_model(trained_rf, X, y)
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_evaluate_model_returns_three_keys(trained_rf, xy):
    """evaluate_model → le dict contient exactement roc_auc, f1, accuracy."""
    X, y = xy
    metrics = evaluate_model(trained_rf, X, y)
    assert set(metrics.keys()) == {"roc_auc", "f1", "accuracy"}


def test_evaluate_model_accuracy_in_range(trained_rf, xy):
    """evaluate_model → accuracy ∈ [0, 1]."""
    X, y = xy
    metrics = evaluate_model(trained_rf, X, y)
    assert 0.0 <= metrics["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# Tests — select_best_model
# ---------------------------------------------------------------------------

def test_select_best_model_returns_highest_auc():
    """select_best_model → retourne le nom avec le plus haut roc_auc."""
    results = {
        "lr":  {"roc_auc": 0.82, "f1": 0.75, "accuracy": 0.80},
        "rf":  {"roc_auc": 0.91, "f1": 0.85, "accuracy": 0.88},
        "svm": {"roc_auc": 0.87, "f1": 0.80, "accuracy": 0.84},
    }
    assert select_best_model(results) == "rf"


def test_select_best_model_name_in_results():
    """select_best_model → le nom retourné est bien une clé du dict."""
    results = {
        "a": {"roc_auc": 0.7},
        "b": {"roc_auc": 0.9},
    }
    best = select_best_model(results)
    assert best in results


# ---------------------------------------------------------------------------
# Tests — save_model / load_model
# ---------------------------------------------------------------------------

def test_save_model_creates_file(trained_rf):
    """save_model → le fichier .joblib existe après sauvegarde."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = save_model(trained_rf, "random_forest", tmpdir)
        assert out.exists()


def test_load_model_returns_pipeline(trained_rf):
    """load_model → retourne un objet Pipeline sklearn."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = save_model(trained_rf, "random_forest", tmpdir)
        loaded = load_model(out)
        assert isinstance(loaded, Pipeline)


def test_load_model_can_predict(trained_rf, xy):
    """load_model → le modèle rechargé peut prédire sans erreur."""
    X, y = xy
    with tempfile.TemporaryDirectory() as tmpdir:
        out = save_model(trained_rf, "random_forest", tmpdir)
        loaded = load_model(out)
        preds = loaded.predict(X)
        assert len(preds) == len(y)
