# tests/test_model.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import joblib
import pytest

def test_model_loading():
    """Vérifie que le modèle et le préprocesseur existent et se chargent."""
    assert os.path.exists('models/preprocessor.pkl'), "preprocessor.pkl manquant"
    model_files = [f for f in os.listdir('models') if f.endswith('.pkl') and f != 'preprocessor.pkl' and f != 'best_model_info.pkl']
    assert len(model_files) > 0, "Aucun modèle trouvé dans models/"
    model = joblib.load(os.path.join('models', model_files[0]))
    preprocessor = joblib.load('models/preprocessor.pkl')
    assert model is not None
    assert preprocessor is not None

def test_model_prediction():
    """Teste une prédiction sur une entrée factice."""
    preprocessor = joblib.load('models/preprocessor.pkl')
    model_files = [f for f in os.listdir('models') if f.endswith('.pkl') and f != 'preprocessor.pkl' and f != 'best_model_info.pkl']
    assert len(model_files) > 0
    model = joblib.load(os.path.join('models', model_files[0]))

    # Créer un exemple à partir du fichier Excel s'il existe
    if os.path.exists('data/raw/dataset.xlsx'):
        df = pd.read_excel('data/raw/dataset.xlsx')
        if 'Diagnosis' in df.columns:
            sample = df.drop(columns=['Diagnosis']).iloc[[0]]
        else:
            sample = df.iloc[[0]]
    else:
        # Récupérer dynamiquement les colonnes attendues par le préprocesseur
        if hasattr(preprocessor, 'feature_names_in_'):
            cols = list(preprocessor.feature_names_in_)
            sample = pd.DataFrame(np.zeros((1, len(cols))), columns=cols)
        else:
            # Fallback
            n_features = getattr(preprocessor, 'n_features_in_', 10)
            sample = pd.DataFrame(np.zeros((1, n_features)))

    sample_processed = preprocessor.transform(sample)
    pred = model.predict(sample_processed)
    proba = model.predict_proba(sample_processed)

    assert pred[0] in [0, 1], "La prédiction doit être 0 ou 1"
    assert 0 <= proba[0][1] <= 1, "La probabilité doit être entre 0 et 1"
