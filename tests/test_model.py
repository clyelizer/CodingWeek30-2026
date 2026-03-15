# tests/test_model.py
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

    # Créer un exemple à partir du fichier Excel s'il existe
    if os.path.exists('data/raw/dataset.xlsx'):
        df = pd.read_excel('data/raw/dataset.xlsx')
        if 'Diagnosis' in df.columns:
            sample = df.drop(columns=['Diagnosis']).iloc[[0]]
        else:
            sample = df.iloc[[0]]
    else:
        # Générer des zéros du bon nombre de features
        n_features = preprocessor.n_features_in_ if hasattr(preprocessor, 'n_features_in_') else 10
        sample = pd.DataFrame(np.zeros((1, n_features)))

    sample_processed = preprocessor.transform(sample)
    pred = model.predict(sample_processed)
    proba = model.predict_proba(sample_processed)

    assert pred[0] in [0, 1], "La prédiction doit être 0 ou 1"
    assert 0 <= proba[0][1] <= 1, "La probabilité doit être entre 0 et 1"
