# tests/test_data_processing.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import tempfile
import pytest

from data_processing import (
    optimize_memory,
    load_and_preprocess,
    validate_target_column,
    infer_feature_groups,
    build_preprocessor,
    split_features_target,
    validate_binary_numeric_target,
)

@pytest.fixture
def sample_excel(tmp_path):
    """Crée un fichier Excel temporaire avec les colonnes de la nouvelle base."""
    df = pd.DataFrame({
        'Age': [10, 12, 8, 15, 9, 11, 14, 13, 7, 16],
        'BMI': [18.0, 20.5, 17.2, 22.0, 18.4, 19.1, 21.6, 20.2, 16.8, 23.0],
        'Sex': ['female', 'male', 'female', 'male', 'female', 'male', 'male', 'female', 'female', 'male'],
        'CRP': [1.2, 3.4, 0.5, 2.1, 1.0, 2.8, 3.1, 0.9, 0.4, 3.7],
        'WBC_Count': [7.7, 8.1, 13.2, 11.4, 9.0, 10.6, 12.1, 8.7, 14.0, 10.2],
        'Diagnosis': [0, 1, 0, 1, 0, 1, 1, 0, 0, 1],
    })
    path = str(tmp_path / "test_dataset.xlsx")
    df.to_excel(path, index=False)
    return path

class TestOptimizeMemory:
    def test_integer_downcast(self):
        df = pd.DataFrame({'A': [1, 2, 3]})
        df_opt = optimize_memory(df)
        assert df_opt['A'].dtype == np.int8

    def test_float_downcast(self):
        df = pd.DataFrame({'B': [1.5, 2.5, 3.5]})
        df_opt = optimize_memory(df)
        assert df_opt['B'].dtype == np.float32

    def test_object_to_category(self):
        df = pd.DataFrame({'C': ['x', 'x', 'x', 'y']})
        df_opt = optimize_memory(df)
        assert str(df_opt['C'].dtype) == 'category'

class TestLoadAndPreprocess:
    def test_shapes(self, sample_excel):
        X_tr, X_te, y_tr, y_te, _ = load_and_preprocess(
            sample_excel, target_col='Diagnosis', test_size=0.25, random_state=42
        )
        assert X_tr.shape[0] + X_te.shape[0] == 10
        assert X_tr.shape[0] == len(y_tr)
        assert X_te.shape[0] == len(y_te)

    def test_output_is_numpy(self, sample_excel):
        X_tr, X_te, _, _, _ = load_and_preprocess(
            sample_excel, target_col='Diagnosis', test_size=0.25, random_state=42
        )
        assert isinstance(X_tr, np.ndarray)
        assert isinstance(X_te, np.ndarray)

    def test_binary_target(self, sample_excel):
        _, _, y_tr, y_te, _ = load_and_preprocess(
            sample_excel, target_col='Diagnosis', test_size=0.25, random_state=42
        )
        assert set(y_tr.unique()).issubset({0, 1})
        assert set(y_te.unique()).issubset({0, 1})

    def test_missing_target_raises(self, sample_excel):
        with pytest.raises(ValueError, match="Colonne cible"):
            load_and_preprocess(sample_excel, target_col='NonExistent')

    def test_preprocessor_saved(self, sample_excel, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        load_and_preprocess(sample_excel, target_col='Diagnosis',
                            test_size=0.25, random_state=42)
        assert os.path.exists('models/preprocessor.pkl')


class TestSingleResponsibilityHelpers:
    def test_validate_target_column_raises_when_missing(self):
        df = pd.DataFrame({'A': [1, 2]})
        with pytest.raises(ValueError, match="Colonne cible"):
            validate_target_column(df, 'Diagnosis')

    def test_split_features_target_returns_expected_parts(self):
        df = pd.DataFrame({'x': [1, 2], 'Diagnosis': [0, 1]})
        X, y = split_features_target(df, 'Diagnosis')
        assert 'Diagnosis' not in X.columns
        assert y.name == 'Diagnosis'

    def test_validate_binary_numeric_target_rejects_non_numeric(self):
        y = pd.Series(['yes', 'no'])
        with pytest.raises(ValueError, match="numérique"):
            validate_binary_numeric_target(y)

    def test_infer_feature_groups_assigns_unknown_to_categorical(self):
        X = pd.DataFrame({
            'Age': [10, 11],
            'Sex': ['female', 'male'],
            'New_Feature': ['a', 'b'],
        })
        num, cat, other = infer_feature_groups(X)
        assert 'Age' in num
        assert 'Sex' in cat
        assert 'New_Feature' in other
        assert 'New_Feature' in cat

    def test_build_preprocessor_contains_expected_transformers(self):
        preprocessor = build_preprocessor(['Age'], ['Sex'])
        names = [name for name, _, _ in preprocessor.transformers]
        assert 'num' in names
        assert 'cat' in names
