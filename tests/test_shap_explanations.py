"""
Tests pour `src/shap_explanations.py`.

Vérifient l'extraction normalisée des valeurs SHAP et le comportement
quand la dépendance `shap` est absente (skip gracieux des plots).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

import shap_explanations as se


def test_extract_shap_values_from_list():
	arr0 = np.zeros((2, 3))
	arr1 = np.ones((2, 3))
	out = se._extract_shap_values([arr0, arr1])
	assert np.array_equal(out, arr1)


def test_extract_shap_values_from_3d_array():
	arr = np.random.randn(4, 5, 2)
	out = se._extract_shap_values(arr)
	assert out.shape == (4, 5)


def test_generate_shap_summary_skips_when_shap_missing(monkeypatch):
	monkeypatch.setattr(se, 'SHAP_AVAILABLE', False)
	model = object()
	X_train = np.zeros((10, 4))
	feature_names = ['a', 'b', 'c', 'd']
	result = se.generate_shap_summary(model, X_train, feature_names)
	assert result['status'] == 'skipped'


def test_plot_waterfall_skips_when_shap_missing(monkeypatch):
	monkeypatch.setattr(se, 'SHAP_AVAILABLE', False)
	model = object()
	X = np.zeros((1, 4))
	result = se.plot_waterfall(model, X, ['a', 'b', 'c', 'd'], X)
	assert result['status'] == 'skipped'
