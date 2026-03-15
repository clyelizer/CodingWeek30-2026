"""
Tests pour `src/evaluate_model.py` : métriques, CV et sauvegarde de la ROC.

Vérifient la forme des métriques retournées, la validation croisée et
que la génération d'artefacts (ROC) fonctionne quand activée.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from evaluate_model import (
	compute_classification_metrics,
	compute_cv_auc,
	maybe_save_roc_curve,
	evaluate_model,
)


def _build_dataset():
	X, y = make_classification(
		n_samples=120,
		n_features=8,
		n_informative=5,
		n_redundant=1,
		random_state=42,
	)
	return X[:90], X[90:], y[:90], y[90:]


def test_compute_classification_metrics_keys_and_ranges():
	y_test = np.array([0, 1, 0, 1])
	y_pred = np.array([0, 1, 0, 0])
	y_proba = np.array([0.1, 0.9, 0.2, 0.4])

	metrics = compute_classification_metrics(y_test, y_pred, y_proba)
	expected_keys = {'ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-score'}
	assert set(metrics.keys()) == expected_keys
	for value in metrics.values():
		assert 0.0 <= value <= 1.0


def test_compute_cv_auc_returns_mean_and_std():
	X_train, _, y_train, _ = _build_dataset()
	model = LogisticRegression(max_iter=1000)
	mean, std = compute_cv_auc(model, X_train, y_train, n_splits=3, random_state=42)
	assert 0.0 <= mean <= 1.0
	assert std >= 0.0


def test_maybe_save_roc_curve_creates_file_when_enabled(tmp_path):
	y_test = np.array([0, 1, 0, 1])
	y_proba = np.array([0.2, 0.8, 0.3, 0.7])
	path = maybe_save_roc_curve(
		y_test,
		y_proba,
		model_name='Test Model',
		output_dir=str(tmp_path),
		enabled=True,
	)
	assert path is not None
	assert os.path.exists(path)


def test_evaluate_model_returns_expected_contract(tmp_path):
	X_train, X_test, y_train, y_test = _build_dataset()
	model = LogisticRegression(max_iter=1000)
	metrics = evaluate_model(
		model,
		X_train,
		y_train,
		X_test,
		y_test,
		model_name='LR',
		save_roc=True,
		output_dir=str(tmp_path),
	)
	assert 'ROC-AUC' in metrics
	assert 'CV ROC-AUC mean' in metrics
	assert 'CV ROC-AUC std' in metrics
