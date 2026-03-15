"""
Tests pour les utilitaires généraux dans `src/utils.py`.

Vérifient le formatage d'en-têtes, la création de répertoires et la reproductibilité
via `set_seed`.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

from utils import format_section, ensure_dirs, set_seed


def test_format_section_output_structure():
    text = format_section('Titre', width=10)
    assert 'Titre' in text
    assert '──────────' in text


def test_ensure_dirs_creates_missing_directories(tmp_path):
    p1 = tmp_path / 'a'
    p2 = tmp_path / 'b' / 'c'
    ensure_dirs(str(p1), str(p2))
    assert p1.exists()
    assert p2.exists()


def test_set_seed_makes_numpy_reproducible():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert np.allclose(a, b)
