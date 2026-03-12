"""
tests/test_data_processing.py
==============================
Paradigme fonctionnel strict : un test = une fonction = une assertion précise.

Chaque test est :
  - Indépendant (pas d'état partagé entre tests)
  - Rapide (données synthétiques en mémoire, pas de fichier externe)
  - Précis (une seule assertion par test, ciblée sur le comportement testé)
"""

import pathlib
import tempfile

import pandas as pd
import pytest

from src.data_processing import (
    BINARY_COLS,
    FEATURE_COLS,
    TARGET_COL,
    encode_binary_columns,
    load_raw_data,
    run_pipeline,
    save_processed_data,
    select_columns,
    split_features_target,
    split_train_test,
)


# ---------------------------------------------------------------------------
# Fixtures — données synthétiques représentatives
# ---------------------------------------------------------------------------

RAW_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "data" / "raw" / "data_finale.xlsx"
)


def _make_minimal_df(n: int = 100) -> pd.DataFrame:
    """Crée un DataFrame minimal valide pour tester le pipeline sans I/O."""
    import numpy as np
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "Lower_Right_Abd_Pain":           rng.choice(["yes", "no"], n),
        "Migratory_Pain":                 rng.choice(["yes", "no"], n),
        "Body_Temperature":               rng.uniform(36, 40, n),
        "WBC_Count":                      rng.uniform(3, 25, n),
        "CRP":                            rng.uniform(0, 200, n),
        "Neutrophil_Percentage":          rng.uniform(30, 90, n),
        "Ipsilateral_Rebound_Tenderness": rng.choice(["yes", "no"], n),
        "Appendix_Diameter":              rng.uniform(4, 15, n),
        "Nausea":                         rng.choice(["yes", "no"], n),
        "Age":                            rng.uniform(1, 18, n),
        "Diagnosis":                      rng.choice([0, 1], n),
        "ColonneInutile":                 rng.integers(0, 10, n),  # à filtrer
    })
    return df


# ---------------------------------------------------------------------------
# Tests unitaires — 1 test = 1 fonction = 1 assertion
# ---------------------------------------------------------------------------

def test_load_raw_data_non_empty():
    """load_raw_data → le DataFrame chargé contient au moins une ligne."""
    df = load_raw_data(RAW_PATH)
    assert len(df) > 0


def test_select_columns_keeps_only_expected_columns():
    """select_columns → seules les colonnes features + cible sont présentes."""
    df = _make_minimal_df()
    result = select_columns(df)
    expected = set(FEATURE_COLS + [TARGET_COL])
    assert set(result.columns) == expected


def test_select_columns_drops_extra_column():
    """select_columns → la colonne parasite 'ColonneInutile' est absente."""
    df = _make_minimal_df()
    result = select_columns(df)
    assert "ColonneInutile" not in result.columns


def test_encode_binary_columns_values_are_0_or_1():
    """encode_binary_columns → les colonnes binaires ne contiennent que 0 et 1."""
    df = _make_minimal_df()
    df_sel = select_columns(df)
    result = encode_binary_columns(df_sel)
    for col in BINARY_COLS:
        assert set(result[col].unique()).issubset({0, 1})


def test_encode_binary_columns_yes_maps_to_1():
    """encode_binary_columns → 'yes' est toujours encodé en 1."""
    df = pd.DataFrame({col: ["yes"] for col in BINARY_COLS} | {
        "Body_Temperature": [37.0],
        "WBC_Count": [10.0],
        "CRP": [5.0],
        "Neutrophil_Percentage": [60.0],
        "Appendix_Diameter": [7.0],
        "Age": [10.0],
        "Diagnosis": [1],
    })
    result = encode_binary_columns(df)
    assert result[BINARY_COLS[0]].iloc[0] == 1


def test_split_features_target_same_length():
    """split_features_target → X et y ont le même nombre de lignes."""
    df = encode_binary_columns(select_columns(_make_minimal_df()))
    X, y = split_features_target(df)
    assert len(X) == len(y)


def test_split_features_target_X_has_correct_columns():
    """split_features_target → X contient exactement les FEATURE_COLS."""
    df = encode_binary_columns(select_columns(_make_minimal_df()))
    X, _ = split_features_target(df)
    assert list(X.columns) == FEATURE_COLS


def test_split_train_test_stratified_ratio():
    """split_train_test → la proportion de positifs dans train et test est proche."""
    df = encode_binary_columns(select_columns(_make_minimal_df(500)))
    X, y = split_features_target(df)
    _, _, y_train, y_test = split_train_test(X, y)
    assert abs(y_train.mean() - y_test.mean()) < 0.05


def test_split_train_test_sizes():
    """split_train_test → test représente environ 20% du total."""
    df = encode_binary_columns(select_columns(_make_minimal_df(500)))
    X, y = split_features_target(df)
    X_train, X_test, _, _ = split_train_test(X, y)
    ratio = len(X_test) / (len(X_train) + len(X_test))
    assert abs(ratio - 0.2) < 0.02


def test_save_processed_data_file_exists():
    """save_processed_data → le fichier .joblib est créé sur le disque."""
    df = encode_binary_columns(select_columns(_make_minimal_df(200)))
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = save_processed_data(X_train, X_test, y_train, y_test, tmpdir)
        assert out.exists()


def test_run_pipeline_produces_output_file():
    """run_pipeline → le fichier processed_data.joblib est produit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = run_pipeline(RAW_PATH, tmpdir)
        assert out.exists()


