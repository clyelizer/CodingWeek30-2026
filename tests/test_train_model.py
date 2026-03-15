import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from sklearn.dummy import DummyClassifier

from train_model import build_results_dataframe, select_best_model


def test_build_results_dataframe_shape_matches_models():
    results = {
        'Model A': {'ROC-AUC': 0.80, 'F1-score': 0.70},
        'Model B': {'ROC-AUC': 0.85, 'F1-score': 0.72},
    }
    df = build_results_dataframe(results)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2
    assert 'ROC-AUC' in df.columns


def test_select_best_model_returns_max_metric_name():
    df = pd.DataFrame(
        {'ROC-AUC': [0.70, 0.90, 0.85], 'F1-score': [0.6, 0.8, 0.75]},
        index=['M1', 'M2', 'M3'],
    )
    assert select_best_model(df, metric='ROC-AUC') == 'M2'


def test_build_models_contains_expected_estimators_contract():
    from train_model import build_models

    models = build_models()
    assert 'Random Forest' in models
    for model in models.values():
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
