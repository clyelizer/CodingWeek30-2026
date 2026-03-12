"""
tests/test_model.py
Paradigme fonctionnel : un test = une fonction = une assertion précise.
Tests du pipeline ML : préparation, entraînement, sauvegarde, évaluation.
Aucun fichier externe requis — tout est synthétique.
"""

import tempfile
import pathlib
import numpy as np
import pandas as pd
import pytest

from src.data_processing import (
    preprocess_pipeline,
    optimize_memory,
    get_feature_columns,
    get_target_column,
    encode_target,
)
from src.train_model import (
    prepare_features,
    split_data,
    train_random_forest,
    train_svm,
    save_model,
    load_model,
)
from src.evaluate_model import (
    compute_metrics,
    compare_models,
    predict_proba_safe,
)


# ---------------------------------------------------------------------------
# Fixtures — données synthétiques
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_df():
    """Dataset synthétique minimal reproduisant la structure réelle (40 patients)."""
    n = 40
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "Age":                           rng.uniform(5, 18, n),
        "BMI":                           rng.uniform(14, 28, n),
        "CRP":                           rng.exponential(20, n),
        "WBC_Count":                     rng.uniform(4, 22, n),
        "Hemoglobin":                    rng.uniform(10, 16, n),
        "Body_Temperature":              rng.uniform(36.2, 39.5, n),
        "RDW":                           rng.uniform(11, 15, n),
        "Appendix_Diameter":             rng.uniform(5.5, 9.5, n),
        "Alvarado_Score":                rng.integers(0, 10, n).astype(float),
        "Paedriatic_Appendicitis_Score": rng.integers(0, 10, n).astype(float),
        "Length_of_Stay":                rng.integers(1, 8, n).astype(float),
        "Weight":                        rng.uniform(20, 70, n),
        "Height":                        rng.uniform(100, 180, n),
        "RBC_Count":                     rng.uniform(3.5, 5.5, n),
        "Thrombocyte_Count":             rng.uniform(150, 400, n),
        "Neutrophil_Percentage":         rng.uniform(40, 90, n),
        "US_Number":                     rng.integers(100, 999, n).astype(float),
        "Segmented_Neutrophils":         rng.uniform(40, 85, n),
        "Sex":           ["male" if i % 2 == 0 else "female" for i in range(n)],
        "Diagnosis":     ["appendicitis" if i < 24 else "no appendicitis" for i in range(n)],
        "Neutrophilia":  ["yes" if i % 3 == 0 else "no" for i in range(n)],
        "Migratory_Pain":    ["yes" if i % 2 == 0 else "no" for i in range(n)],
        "Lower_Right_Abd_Pain": ["yes"] * n,
        "Contralateral_Rebound_Tenderness": ["no"] * n,
        "Coughing_Pain": ["yes" if i % 3 == 0 else "no" for i in range(n)],
        "Nausea":        ["yes" if i % 2 == 0 else "no" for i in range(n)],
        "Loss_of_Appetite": ["yes" if i % 4 == 0 else "no" for i in range(n)],
        "Dysuria":       ["no"] * n,
        "Stool":         ["normal" if i % 3 == 0 else "constipation" for i in range(n)],
        "Peritonitis":   ["no"] * n,
        "Psoas_Sign":    ["no"] * n,
        "Appendix_on_US": ["yes" if i % 2 == 0 else "no" for i in range(n)],
        "US_Performed":  ["yes"] * n,
        "Free_Fluids":   ["no"] * n,
        "Diagnosis_Presumptive": ["appendicitis"] * 24 + ["no appendicitis"] * 16,
        "RBC_in_Urine":  ["no"] * n,
        "Ketones_in_Urine": ["no"] * n,
        "WBC_in_Urine":  ["no"] * n,
        "Ipsilateral_Rebound_Tenderness": ["yes" if i % 2 == 0 else "no" for i in range(n)],
        "Abscess_Location":            [np.nan] * n,
        "Gynecological_Findings":      [np.nan] * n,
        "Conglomerate_of_Bowel_Loops": [np.nan] * n,
        "Ileus":                       [np.nan] * (n - 2) + ["yes", "no"],
        "Perfusion":                   [np.nan] * n,
        "Enteritis":                   [np.nan] * n,
        "Appendicolith":               [np.nan] * n,
        "Coprostasis":                 [np.nan] * n,
        "Perforation":                 [np.nan] * n,
        "Appendicular_Abscess":        [np.nan] * n,
        "Bowel_Wall_Thickening":       [np.nan] * n,
        "Lymph_Nodes_Location":        [np.nan] * n,
        "Target_Sign":                 [np.nan] * n,
        "Meteorism":                   [np.nan] * n,
        "Pathological_Lymph_Nodes":    [np.nan] * n,
        "Appendix_Wall_Layers":        [np.nan] * n,
        "Surrounding_Tissue_Reaction": [np.nan] * n,
    })


@pytest.fixture(scope="module")
def processed_dataset(synthetic_df):
    """Dataset après pipeline de préprocessing."""
    return preprocess_pipeline(synthetic_df)


@pytest.fixture(scope="module")
def X_y(processed_dataset):
    """Features et cible prêtes pour l'entraînement."""
    feature_cols = get_feature_columns(processed_dataset)
    target_col = get_target_column()
    return prepare_features(processed_dataset, feature_cols, target_col)


@pytest.fixture(scope="module")
def train_test(X_y):
    """Split train/test."""
    X, y = X_y
    return split_data(X, y, test_size=0.25, random_state=42)


@pytest.fixture(scope="module")
def trained_rf(train_test):
    """Random Forest entraîné sur les données synthétiques."""
    X_train, _, y_train, _ = train_test
    return train_random_forest(X_train, y_train, n_estimators=10)


@pytest.fixture(scope="module")
def trained_svm(train_test):
    """SVM entraîné sur les données synthétiques."""
    X_train, _, y_train, _ = train_test
    return train_svm(X_train, y_train, C=1.0)


# ---------------------------------------------------------------------------
# Tests — prepare_features
# ---------------------------------------------------------------------------

def test_prepare_features_X_is_numeric(processed_dataset):
    """X ne doit contenir que des colonnes numériques après préparation."""
    feature_cols = get_feature_columns(processed_dataset)
    target_col = get_target_column()
    X, _ = prepare_features(processed_dataset, feature_cols, target_col)
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    assert len(non_numeric) == 0, f"Colonnes non-numériques restantes : {non_numeric}"


def test_prepare_features_y_is_binary(processed_dataset):
    """y doit contenir uniquement 0 et 1."""
    feature_cols = get_feature_columns(processed_dataset)
    target_col = get_target_column()
    _, y = prepare_features(processed_dataset, feature_cols, target_col)
    assert set(y.unique()).issubset({0, 1})


def test_prepare_features_no_target_in_X(processed_dataset):
    """La colonne cible ne doit pas apparaître dans X."""
    feature_cols = get_feature_columns(processed_dataset)
    target_col = get_target_column()
    X, _ = prepare_features(processed_dataset, feature_cols, target_col)
    assert target_col not in X.columns


def test_prepare_features_lengths_match(processed_dataset):
    """X et y doivent avoir le même nombre de lignes."""
    feature_cols = get_feature_columns(processed_dataset)
    target_col = get_target_column()
    X, y = prepare_features(processed_dataset, feature_cols, target_col)
    assert len(X) == len(y)


# ---------------------------------------------------------------------------
# Tests — split_data
# ---------------------------------------------------------------------------

def test_split_data_correct_test_size(X_y):
    """La taille du test set doit respecter le ratio demandé (± 1 ligne)."""
    X, y = X_y
    X_tr, X_te, _, _ = split_data(X, y, test_size=0.25)
    expected_test = int(len(X) * 0.25)
    assert abs(len(X_te) - expected_test) <= 1


def test_split_data_no_overlap_between_train_and_test(X_y):
    """Les index train et test ne doivent pas se chevaucher."""
    X, y = X_y
    X_tr, X_te, _, _ = split_data(X, y)
    train_idx = set(X_tr.index)
    test_idx = set(X_te.index)
    assert len(train_idx & test_idx) == 0


def test_split_data_total_rows_preserved(X_y):
    """Train + test = dataset complet."""
    X, y = X_y
    X_tr, X_te, y_tr, y_te = split_data(X, y)
    assert len(X_tr) + len(X_te) == len(X)
    assert len(y_tr) + len(y_te) == len(y)


# ---------------------------------------------------------------------------
# Tests — train_random_forest
# ---------------------------------------------------------------------------

def test_train_rf_returns_fitted_model(train_test):
    """Le modèle doit avoir l'attribut 'estimators_' après fit."""
    X_train, _, y_train, _ = train_test
    model = train_random_forest(X_train, y_train, n_estimators=5)
    assert hasattr(model, "estimators_")


def test_train_rf_predict_shape(train_test):
    """Les prédictions doivent avoir la même longueur que X_test."""
    X_train, X_test, y_train, _ = train_test
    model = train_random_forest(X_train, y_train, n_estimators=5)
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)


def test_train_rf_predict_proba_shape(train_test):
    """predict_proba doit retourner (n_samples, 2)."""
    X_train, X_test, y_train, _ = train_test
    model = train_random_forest(X_train, y_train, n_estimators=5)
    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)


def test_train_rf_predict_only_0_or_1(train_test):
    """Les prédictions doivent être binaires."""
    X_train, X_test, y_train, _ = train_test
    model = train_random_forest(X_train, y_train, n_estimators=5)
    preds = model.predict(X_test)
    assert set(preds).issubset({0, 1})


# ---------------------------------------------------------------------------
# Tests — train_svm
# ---------------------------------------------------------------------------

def test_train_svm_returns_pipeline(trained_svm):
    """SVM retourne un Pipeline sklearn."""
    from sklearn.pipeline import Pipeline
    assert isinstance(trained_svm, Pipeline)


def test_train_svm_has_predict_proba(train_test, trained_svm):
    """Le modèle SVM entraîné doit supporter predict_proba."""
    _, X_test, _, _ = train_test
    proba = trained_svm.predict_proba(X_test)
    assert proba.shape[1] == 2


# ---------------------------------------------------------------------------
# Tests — save_model / load_model
# ---------------------------------------------------------------------------

def test_save_and_load_model_random_forest(trained_rf, train_test):
    """Sauvegarder puis charger un RF produit le même résultat de prédiction."""
    _, X_test, _, _ = train_test
    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "rf.joblib"
        save_model(trained_rf, path)
        loaded = load_model(path)
    preds_orig = trained_rf.predict(X_test)
    preds_load = loaded.predict(X_test)
    assert np.array_equal(preds_orig, preds_load)


def test_save_model_creates_file(trained_rf):
    """save_model doit créer le fichier .joblib."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "subdir" / "rf.joblib"
        save_model(trained_rf, path)
        assert path.exists()


def test_load_model_from_disk_has_predict(trained_rf, train_test):
    """Le modèle chargé depuis le disque doit avoir la méthode predict."""
    _, X_test, _, _ = train_test
    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "model.joblib"
        save_model(trained_rf, path)
        model = load_model(path)
    assert hasattr(model, "predict")


# ---------------------------------------------------------------------------
# Tests — compute_metrics
# ---------------------------------------------------------------------------

def test_compute_metrics_keys(trained_rf, train_test):
    """compute_metrics doit retourner exactement les 5 métriques attendues."""
    _, X_test, _, y_test = train_test
    metrics = compute_metrics(trained_rf, X_test, y_test)
    expected_keys = {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert set(metrics.keys()) == expected_keys


def test_compute_metrics_values_in_range(trained_rf, train_test):
    """Toutes les métriques doivent être dans [0, 1]."""
    _, X_test, _, y_test = train_test
    metrics = compute_metrics(trained_rf, X_test, y_test)
    for name, val in metrics.items():
        assert 0.0 <= val <= 1.0, f"{name} = {val} hors intervalle [0, 1]"


def test_compute_metrics_roc_auc_above_threshold(trained_rf, train_test):
    """Sur données synthétiques équilibrées, le RF doit dépasser 0.5 en AUC."""
    _, X_test, _, y_test = train_test
    metrics = compute_metrics(trained_rf, X_test, y_test)
    assert metrics["roc_auc"] >= 0.5, f"AUC = {metrics['roc_auc']} ≤ 0.5 (pire qu'aléatoire)"


# ---------------------------------------------------------------------------
# Tests — compare_models
# ---------------------------------------------------------------------------

def test_compare_models_returns_dataframe(trained_rf, trained_svm, train_test):
    """compare_models doit retourner un DataFrame."""
    _, X_test, _, y_test = train_test
    models = {"random_forest": trained_rf, "svm": trained_svm}
    result = compare_models(models, X_test, y_test)
    assert isinstance(result, pd.DataFrame)


def test_compare_models_sorted_by_roc_auc(trained_rf, trained_svm, train_test):
    """Le résultat doit être trié par roc_auc décroissant."""
    _, X_test, _, y_test = train_test
    models = {"random_forest": trained_rf, "svm": trained_svm}
    result = compare_models(models, X_test, y_test)
    auc_values = result["roc_auc"].tolist()
    assert auc_values == sorted(auc_values, reverse=True)


def test_compare_models_one_row_per_model(trained_rf, trained_svm, train_test):
    """Une ligne par modèle dans le résultat."""
    _, X_test, _, y_test = train_test
    models = {"random_forest": trained_rf, "svm": trained_svm}
    result = compare_models(models, X_test, y_test)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Tests — predict_proba_safe
# ---------------------------------------------------------------------------

def test_predict_proba_safe_rf_returns_1d(trained_rf, train_test):
    """predict_proba_safe doit retourner un array 1D pour RF."""
    _, X_test, _, _ = train_test
    proba = predict_proba_safe(trained_rf, X_test)
    assert proba.ndim == 1
    assert len(proba) == len(X_test)


def test_predict_proba_safe_values_between_0_and_1(trained_rf, train_test):
    """Les probabilités doivent être dans [0, 1]."""
    _, X_test, _, _ = train_test
    proba = predict_proba_safe(trained_rf, X_test)
    assert (proba >= 0).all() and (proba <= 1).all()


# ---------------------------------------------------------------------------
# Test d'intégration — optimize_memory améliore toujours la mémoire
# ---------------------------------------------------------------------------

def test_optimize_memory_always_reduces_on_processed_data(processed_dataset):
    """
    Test d'intégration : optimize_memory appliqué au dataset réel traité
    doit réduire la consommation mémoire.
    """
    before = processed_dataset.memory_usage(deep=True).sum()
    optimized = optimize_memory(processed_dataset)
    after = optimized.memory_usage(deep=True).sum()
    reduction_pct = (1 - after / before) * 100
    assert reduction_pct > 0, f"Aucune réduction mémoire : {reduction_pct:.1f}%"
    print(f"\n  Réduction mémoire : {reduction_pct:.1f}%  ({before/1024:.1f}KB → {after/1024:.1f}KB)")
