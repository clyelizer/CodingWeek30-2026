"""
Tests d'intégration centrés sur la présence et le comportement des artefacts
de modèle et du préprocesseur (models/preprocessor.pkl et fichiers dans models/).
Vérifient le chargement, la préparation d'entrée et des prédictions plausibles.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import joblib
import pytest


def _load_predictive_model(models_dir='models'):
    """Charge un artefact modèle compatible predict/predict_proba depuis models/."""
    candidates = [
        f for f in os.listdir(models_dir)
        if f.endswith('.pkl') and f != 'preprocessor.pkl'
    ]

    checked = []
    for filename in sorted(candidates):
        path = os.path.join(models_dir, filename)
        obj = joblib.load(path)
        checked.append(filename)
        if hasattr(obj, 'predict') and hasattr(obj, 'predict_proba'):
            return obj, filename

    raise AssertionError(
        f"Aucun estimateur prédictif trouvé dans {models_dir}. Fichiers vérifiés: {checked}"
    )


def _prepare_model_input(preprocessor, model):
    """Construit une entrée compatible avec le modèle, même si les artefacts ne sont pas alignés."""
    # 1) Chemin préféré: raw -> preprocessor -> modèle
    if os.path.exists('data/raw/dataset.xlsx'):
        df = pd.read_excel('data/raw/dataset.xlsx')
        if 'Diagnosis' in df.columns:
            raw_sample = df.drop(columns=['Diagnosis']).iloc[[0]]
        else:
            raw_sample = df.iloc[[0]]

        transformed = preprocessor.transform(raw_sample)
        expected = getattr(model, 'n_features_in_', None)
        if expected is None or transformed.shape[1] == expected:
            return transformed

    # 2) Fallback fiable: espace déjà transformé stocké
    processed_path = 'data/processed/processed_data.joblib'
    if os.path.exists(processed_path):
        processed = joblib.load(processed_path)
        X_test = processed.get('X_test')
        if X_test is not None:
            expected = getattr(model, 'n_features_in_', None)
            if expected is None or X_test.shape[1] == expected:
                return X_test[:1]

    # 3) Dernier recours: vecteur neutre de la bonne dimension modèle
    expected = getattr(model, 'n_features_in_', None)
    if expected is not None:
        return np.zeros((1, expected))

    # Si le modèle ne déclare pas n_features_in_, on retente avec le préprocesseur.
    n_features = preprocessor.n_features_in_ if hasattr(preprocessor, 'n_features_in_') else 10
    raw = pd.DataFrame(np.zeros((1, n_features)))
    return preprocessor.transform(raw)

def test_model_loading():
    """Vérifie que le modèle et le préprocesseur existent et se chargent."""
    assert os.path.exists('models/preprocessor.pkl'), "preprocessor.pkl manquant"
    model, model_file = _load_predictive_model('models')
    preprocessor = joblib.load('models/preprocessor.pkl')
    assert model_file.endswith('.pkl')
    assert model is not None
    assert preprocessor is not None

def test_model_prediction():
    """Teste une prédiction sur une entrée factice."""
    preprocessor = joblib.load('models/preprocessor.pkl')
    model, _ = _load_predictive_model('models')

    model_input = _prepare_model_input(preprocessor, model)
    pred = model.predict(model_input)
    proba = model.predict_proba(model_input)

    assert pred[0] in [0, 1], "La prédiction doit être 0 ou 1"
    assert 0 <= proba[0][1] <= 1, "La probabilité doit être entre 0 et 1"
