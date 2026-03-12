"""
tests/test_data_processing.py
Paradigme fonctionnel : un test = une fonction = une assertion précise.
Chaque test est indépendant, rapide, sans fichier externe.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_processing import (
    impute_numeric,
    impute_categorical,
    encode_sparse_binary,
    remove_biological_impossibles,
    winsorize_iqr,
    add_log_transform,
    optimize_memory,
    get_feature_columns,
    encode_target,
    preprocess_pipeline,
)


# ---------------------------------------------------------------------------
# Fixtures — datasets synthétiques minimaux
# ---------------------------------------------------------------------------

@pytest.fixture
def df_numeric_with_na():
    """DataFrame avec 2 colonnes numériques contenant des NA."""
    return pd.DataFrame({
        "Age":       [10.0, np.nan, 12.0, np.nan, 14.0],
        "WBC_Count": [8.0,  12.0,  np.nan, 9.0,  7.0],
    })


@pytest.fixture
def df_categorical_with_na():
    """DataFrame avec 1 colonne catégorielle contenant des NA."""
    return pd.DataFrame({
        "Sex": ["male", "female", np.nan, "male", "female"],
    })


@pytest.fixture
def df_sparse_cols():
    """DataFrame avec colonne sparse (98% NA en production)."""
    return pd.DataFrame({
        "Abscess_Location": [np.nan, "fossa iliaca", np.nan, np.nan, "pelvis"],
        "Score":            [1, 2, 3, 4, 5],
    })


@pytest.fixture
def df_bio_impossible():
    """DataFrame avec valeurs biologiquement impossibles — une violation par ligne."""
    return pd.DataFrame({
        #                  row0 (low temp)  row1 (high Hgb)  row2 (high RDW)  row3 (high RBC)
        "Body_Temperature": [33.0,           37.0,            37.0,            37.0],
        "Hemoglobin":       [13.0,           25.0,            13.0,            13.0],
        "RDW":              [12.5,           13.0,            31.0,            12.0],
        "RBC_Count":        [4.5,            4.8,             4.2,             9.1],
    })


@pytest.fixture
def df_for_winsor():
    """DataFrame avec outliers extrêmes sur BMI."""
    return pd.DataFrame({
        "BMI": [18.0, 19.0, 20.0, 21.0, 100.0],   # 100 est un outlier extrême
    })


@pytest.fixture
def df_crp():
    """DataFrame avec colonne CRP pour log-transformation."""
    return pd.DataFrame({"CRP": [0.0, 1.0, 10.0, 100.0, 365.0]})


@pytest.fixture
def df_memory():
    """DataFrame avec types sous-optimaux (int64, float64, object)."""
    return pd.DataFrame({
        "binaire":   pd.array([0, 1, 0, 1, 1], dtype="int64"),
        "score":     pd.array([1, 2, 3, 4, 5], dtype="int64"),
        "mesure":    pd.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype="float64"),
        "sexe":      pd.array(["M", "F", "M", "F", "M"], dtype="object"),
    })


@pytest.fixture
def df_full():
    """Dataset synthétique complet simulant le dataset réel (colonnes minimales)."""
    n = 20
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "Age":                           rng.uniform(5, 18, n),
        "BMI":                           rng.uniform(14, 28, n),
        "CRP":                           rng.exponential(20, n),
        "WBC_Count":                     rng.uniform(4, 25, n),
        "Hemoglobin":                    rng.uniform(10, 16, n),
        "Body_Temperature":              rng.uniform(36.0, 39.5, n),
        "RDW":                           rng.uniform(11, 16, n),
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
        "RDW":                           rng.uniform(11, 15, n),
        "Sex":                           ["male" if i % 2 == 0 else "female" for i in range(n)],
        "Diagnosis":                     ["appendicitis" if i < 12 else "no appendicitis" for i in range(n)],
        "Neutrophilia":                  ["yes" if i % 3 == 0 else "no" for i in range(n)],
        "Migratory_Pain":                ["yes" if i % 2 == 0 else "no" for i in range(n)],
        "Lower_Right_Abd_Pain":          ["yes"] * n,
        "Contralateral_Rebound_Tenderness": ["no"] * n,
        "Coughing_Pain":                 ["yes" if i % 3 == 0 else "no" for i in range(n)],
        "Nausea":                        ["yes" if i % 2 == 0 else "no" for i in range(n)],
        "Loss_of_Appetite":              ["yes" if i % 4 == 0 else "no" for i in range(n)],
        "Dysuria":                       ["no"] * n,
        "Stool":                         ["normal" if i % 3 == 0 else "constipation" for i in range(n)],
        "Peritonitis":                   ["no"] * n,
        "Psoas_Sign":                    ["no"] * n,
        "Appendix_on_US":               ["yes" if i % 2 == 0 else "no" for i in range(n)],
        "US_Performed":                  ["yes"] * n,
        "Free_Fluids":                   ["no"] * n,
        "Diagnosis_Presumptive":         ["appendicitis"] * 12 + ["no appendicitis"] * 8,
        "Abscess_Location":              [np.nan] * n,
        "Gynecological_Findings":        [np.nan] * n,
        "Conglomerate_of_Bowel_Loops":   [np.nan] * n,
        "Ileus":                         [np.nan] * (n - 2) + ["yes", "no"],
        "Perfusion":                     [np.nan] * n,
        "Enteritis":                     [np.nan] * n,
        "Appendicolith":                 [np.nan] * n,
        "Coprostasis":                   [np.nan] * n,
        "Perforation":                   [np.nan] * n,
        "Appendicular_Abscess":          [np.nan] * n,
        "Bowel_Wall_Thickening":         [np.nan] * n,
        "Lymph_Nodes_Location":          [np.nan] * n,
        "Target_Sign":                   [np.nan] * n,
        "Meteorism":                     [np.nan] * n,
        "Pathological_Lymph_Nodes":      [np.nan] * n,
        "Appendix_Wall_Layers":          [np.nan] * n,
        "Surrounding_Tissue_Reaction":   [np.nan] * n,
        "RBC_in_Urine":                  ["no"] * n,
        "Ketones_in_Urine":              ["no"] * n,
        "WBC_in_Urine":                  ["no"] * n,
        "Ipsilateral_Rebound_Tenderness": ["yes" if i % 2 == 0 else "no" for i in range(n)],
    })


# ---------------------------------------------------------------------------
# Tests — impute_numeric
# ---------------------------------------------------------------------------

def test_impute_numeric_removes_all_na(df_numeric_with_na):
    """Après imputation, aucune valeur manquante dans les colonnes traitées."""
    result = impute_numeric(df_numeric_with_na, ["Age", "WBC_Count"])
    assert result["Age"].isna().sum() == 0
    assert result["WBC_Count"].isna().sum() == 0


def test_impute_numeric_uses_median(df_numeric_with_na):
    """La valeur imputée doit être égale à la médiane des valeurs non-nulles."""
    result = impute_numeric(df_numeric_with_na, ["Age"])
    expected_median = df_numeric_with_na["Age"].median()
    imputed_values = result.loc[df_numeric_with_na["Age"].isna(), "Age"]
    assert (imputed_values == expected_median).all()


def test_impute_numeric_does_not_modify_original(df_numeric_with_na):
    """La fonction ne modifie pas le DataFrame d'entrée (pas d'effet de bord)."""
    original_na_count = df_numeric_with_na["Age"].isna().sum()
    _ = impute_numeric(df_numeric_with_na, ["Age"])
    assert df_numeric_with_na["Age"].isna().sum() == original_na_count


def test_impute_numeric_ignores_missing_columns(df_numeric_with_na):
    """Les colonnes absentes du DataFrame sont silencieusement ignorées."""
    result = impute_numeric(df_numeric_with_na, ["Age", "ColonneInexistante"])
    assert "ColonneInexistante" not in result.columns


def test_impute_numeric_mean_strategy(df_numeric_with_na):
    """La stratégie 'mean' impute par la moyenne."""
    result = impute_numeric(df_numeric_with_na, ["Age"], strategy="mean")
    expected_mean = df_numeric_with_na["Age"].mean()
    imputed_values = result.loc[df_numeric_with_na["Age"].isna(), "Age"]
    assert np.allclose(imputed_values.values, expected_mean)


# ---------------------------------------------------------------------------
# Tests — impute_categorical
# ---------------------------------------------------------------------------

def test_impute_categorical_removes_all_na(df_categorical_with_na):
    """Après imputation, aucune valeur manquante dans la colonne traitée."""
    result = impute_categorical(df_categorical_with_na, ["Sex"])
    assert result["Sex"].isna().sum() == 0


def test_impute_categorical_uses_mode(df_categorical_with_na):
    """La valeur imputée doit être la mode de la colonne."""
    expected_mode = df_categorical_with_na["Sex"].mode()[0]
    result = impute_categorical(df_categorical_with_na, ["Sex"])
    imputed_val = result.loc[df_categorical_with_na["Sex"].isna(), "Sex"].iloc[0]
    assert imputed_val == expected_mode


def test_impute_categorical_does_not_modify_original(df_categorical_with_na):
    """La fonction ne modifie pas le DataFrame d'entrée."""
    original_na = df_categorical_with_na["Sex"].isna().sum()
    _ = impute_categorical(df_categorical_with_na, ["Sex"])
    assert df_categorical_with_na["Sex"].isna().sum() == original_na


# ---------------------------------------------------------------------------
# Tests — encode_sparse_binary
# ---------------------------------------------------------------------------

def test_encode_sparse_binary_values_are_0_or_1(df_sparse_cols):
    """Après encodage, la colonne ne contient que 0 et 1."""
    result = encode_sparse_binary(df_sparse_cols, ["Abscess_Location"])
    unique_vals = set(result["Abscess_Location"].unique())
    assert unique_vals.issubset({0, 1})


def test_encode_sparse_binary_na_becomes_0(df_sparse_cols):
    """Les NA originaux deviennent 0."""
    result = encode_sparse_binary(df_sparse_cols, ["Abscess_Location"])
    na_mask = df_sparse_cols["Abscess_Location"].isna()
    assert (result.loc[na_mask, "Abscess_Location"] == 0).all()


def test_encode_sparse_binary_present_becomes_1(df_sparse_cols):
    """Les valeurs présentes deviennent 1."""
    result = encode_sparse_binary(df_sparse_cols, ["Abscess_Location"])
    present_mask = df_sparse_cols["Abscess_Location"].notna()
    assert (result.loc[present_mask, "Abscess_Location"] == 1).all()


def test_encode_sparse_binary_dtype_is_int(df_sparse_cols):
    """La colonne encodée doit être de type numérique entier."""
    result = encode_sparse_binary(df_sparse_cols, ["Abscess_Location"])
    assert pd.api.types.is_integer_dtype(result["Abscess_Location"])


# ---------------------------------------------------------------------------
# Tests — remove_biological_impossibles
# ---------------------------------------------------------------------------

def test_remove_biological_impossibles_removes_low_temp(df_bio_impossible):
    """Lignes avec Body_Temperature ≤ 34 doivent être supprimées."""
    result = remove_biological_impossibles(df_bio_impossible)
    assert (result["Body_Temperature"] > 34).all()


def test_remove_biological_impossibles_removes_high_hemoglobin(df_bio_impossible):
    """Lignes avec Hemoglobin > 20 doivent être supprimées."""
    result = remove_biological_impossibles(df_bio_impossible)
    assert (result["Hemoglobin"] <= 20).all()


def test_remove_biological_impossibles_removes_high_rdw(df_bio_impossible):
    """Lignes avec RDW > 30 doivent être supprimées."""
    result = remove_biological_impossibles(df_bio_impossible)
    assert (result["RDW"] <= 30).all()


def test_remove_biological_impossibles_removes_high_rbc(df_bio_impossible):
    """Lignes avec RBC_Count > 8 doivent être supprimées."""
    result = remove_biological_impossibles(df_bio_impossible)
    assert (result["RBC_Count"] <= 8).all()


def test_remove_biological_impossibles_exact_row_count(df_bio_impossible):
    """
    Chaque ligne du fixture viole exactement une règle différente.
    Après filtrage, toutes les 4 lignes doivent être supprimées.
    """
    result = remove_biological_impossibles(df_bio_impossible)
    assert len(result) == 0


def test_remove_biological_impossibles_does_not_modify_original(df_bio_impossible):
    """La fonction ne modifie pas le DataFrame d'entrée."""
    original_len = len(df_bio_impossible)
    _ = remove_biological_impossibles(df_bio_impossible)
    assert len(df_bio_impossible) == original_len


# ---------------------------------------------------------------------------
# Tests — winsorize_iqr
# ---------------------------------------------------------------------------

def test_winsorize_iqr_caps_upper_outlier(df_for_winsor):
    """La valeur 100 doit être plafonnée après winsorisation."""
    result = winsorize_iqr(df_for_winsor, ["BMI"])
    Q1 = df_for_winsor["BMI"].quantile(0.25)
    Q3 = df_for_winsor["BMI"].quantile(0.75)
    upper_bound = Q3 + 1.5 * (Q3 - Q1)
    assert result["BMI"].max() <= upper_bound + 1e-9


def test_winsorize_iqr_does_not_modify_original(df_for_winsor):
    """La fonction ne modifie pas le DataFrame d'entrée."""
    original_max = df_for_winsor["BMI"].max()
    _ = winsorize_iqr(df_for_winsor, ["BMI"])
    assert df_for_winsor["BMI"].max() == original_max


def test_winsorize_iqr_preserves_len(df_for_winsor):
    """Winsorisation ne supprime pas de lignes."""
    result = winsorize_iqr(df_for_winsor, ["BMI"])
    assert len(result) == len(df_for_winsor)


# ---------------------------------------------------------------------------
# Tests — add_log_transform
# ---------------------------------------------------------------------------

def test_add_log_transform_creates_column(df_crp):
    """La colonne CRP_log doit être créée."""
    result = add_log_transform(df_crp, "CRP")
    assert "CRP_log" in result.columns


def test_add_log_transform_values_correct(df_crp):
    """Les valeurs doivent être égales à log1p(CRP)."""
    result = add_log_transform(df_crp, "CRP")
    expected = np.log1p(df_crp["CRP"].values)
    assert np.allclose(result["CRP_log"].values, expected)


def test_add_log_transform_no_negatives(df_crp):
    """log1p(x) avec x ≥ 0 ne doit jamais produire de valeurs négatives."""
    result = add_log_transform(df_crp, "CRP")
    assert (result["CRP_log"] >= 0).all()


def test_add_log_transform_custom_name(df_crp):
    """Le nom de la colonne créée est paramétrable."""
    result = add_log_transform(df_crp, "CRP", new_col="log_crp_custom")
    assert "log_crp_custom" in result.columns


# ---------------------------------------------------------------------------
# Tests — optimize_memory
# ---------------------------------------------------------------------------

def test_optimize_memory_reduces_size(df_memory):
    """La mémoire après optimisation doit être strictement inférieure à avant."""
    before = df_memory.memory_usage(deep=True).sum()
    result = optimize_memory(df_memory)
    after = result.memory_usage(deep=True).sum()
    assert after < before, f"Pas de réduction : {before} → {after}"


def test_optimize_memory_converts_float64_to_float32(df_memory):
    """Les colonnes float64 doivent être converties en float32."""
    result = optimize_memory(df_memory)
    assert result["mesure"].dtype == np.float32


def test_optimize_memory_converts_low_int_to_int8(df_memory):
    """Une colonne binaire (0/1) doit être convertie en int8."""
    result = optimize_memory(df_memory)
    assert result["binaire"].dtype == np.int8


def test_optimize_memory_converts_object_to_category(df_memory):
    """Les colonnes object à faible cardinalité doivent devenir category."""
    result = optimize_memory(df_memory)
    assert result["sexe"].dtype.name == "category"


def test_optimize_memory_does_not_modify_original(df_memory):
    """La fonction ne modifie pas le DataFrame d'entrée."""
    original_dtype = df_memory["mesure"].dtype
    _ = optimize_memory(df_memory)
    assert df_memory["mesure"].dtype == original_dtype


def test_optimize_memory_preserves_values(df_memory):
    """Les valeurs numériques doivent être préservées malgré la conversion."""
    result = optimize_memory(df_memory)
    assert np.allclose(result["mesure"].astype(float).values,
                       df_memory["mesure"].astype(float).values, atol=1e-3)


# ---------------------------------------------------------------------------
# Tests — encode_target
# ---------------------------------------------------------------------------

def test_encode_target_appendicitis_is_1():
    """'appendicitis' doit être encodé en 1."""
    s = pd.Series(["appendicitis", "no appendicitis", "appendicitis"])
    result = encode_target(s)
    assert result.tolist() == [1, 0, 1]


def test_encode_target_non_appendicitis_is_0():
    """Toute valeur autre que 'appendicitis' doit être 0."""
    s = pd.Series(["no appendicitis", "gastroenteritis", "appendicitis"])
    result = encode_target(s)
    assert result.tolist() == [0, 0, 1]


def test_encode_target_dtype_is_int():
    """La série résultante doit être de type entier."""
    s = pd.Series(["appendicitis", "no appendicitis"])
    result = encode_target(s)
    assert pd.api.types.is_integer_dtype(result)


# ---------------------------------------------------------------------------
# Tests — get_feature_columns
# ---------------------------------------------------------------------------

def test_get_feature_columns_excludes_target(df_full):
    """La colonne Diagnosis ne doit pas apparaître dans les features."""
    df_processed = preprocess_pipeline(df_full)
    features = get_feature_columns(df_processed)
    assert "Diagnosis" not in features


def test_get_feature_columns_excludes_leakage(df_full):
    """Les colonnes de data leakage ne doivent pas apparaître dans les features."""
    df_processed = preprocess_pipeline(df_full)
    features = get_feature_columns(df_processed)
    leakage = ["Severity", "Management", "Diagnosis_Presumptive",
               "Alvarado_Score", "Paedriatic_Appendicitis_Score"]
    for col in leakage:
        if col in df_processed.columns:
            assert col not in features, f"{col} est dans les features (leakage!)"


def test_get_feature_columns_returns_list(df_full):
    """La fonction retourne bien une liste."""
    df_processed = preprocess_pipeline(df_full)
    features = get_feature_columns(df_processed)
    assert isinstance(features, list)
    assert len(features) > 0


# ---------------------------------------------------------------------------
# Tests — preprocess_pipeline
# ---------------------------------------------------------------------------

def test_preprocess_pipeline_no_missing_values(df_full):
    """Après le pipeline complet, aucune valeur manquante ne doit subsister."""
    result = preprocess_pipeline(df_full)
    total_na = result.isnull().sum().sum()
    assert total_na == 0, f"{total_na} valeurs manquantes restantes"


def test_preprocess_pipeline_adds_crp_log(df_full):
    """Le pipeline doit créer la colonne CRP_log."""
    result = preprocess_pipeline(df_full)
    assert "CRP_log" in result.columns


def test_preprocess_pipeline_crp_log_non_negative(df_full):
    """CRP_log = log1p(CRP) ≥ 0 pour CRP ≥ 0."""
    result = preprocess_pipeline(df_full)
    assert (result["CRP_log"] >= 0).all()


def test_preprocess_pipeline_binary_cols_are_0_or_1(df_full):
    """Les colonnes éparses encodées ne doivent contenir que 0 et 1."""
    from src.data_processing import SPARSE_BINARY_COLS
    result = preprocess_pipeline(df_full)
    for col in SPARSE_BINARY_COLS:
        if col in result.columns:
            unique = set(result[col].unique())
            assert unique.issubset({0, 1}), f"{col} contient {unique}"


def test_preprocess_pipeline_preserves_diagnosis_column(df_full):
    """La colonne Diagnosis doit être conservée (nécessaire pour encode_target)."""
    result = preprocess_pipeline(df_full)
    assert "Diagnosis" in result.columns
