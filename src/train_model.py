# src/train_model.py
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import joblib
import json

from sklearn.ensemble import RandomForestClassifier
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    from .data_processing import load_and_preprocess, get_feature_names
    from .evaluate_model import evaluate_model
except ImportError:
    from data_processing import load_and_preprocess, get_feature_names
    from evaluate_model import evaluate_model

# Configuration
FILEPATH     = 'data/raw/dataset.xlsx'
TARGET_COL   = 'Diagnosis'
TEST_SIZE    = 0.2
RANDOM_STATE = 42


def load_model(model_path='models/Random_Forest.pkl'):
    """Charge un modèle depuis un chemin explicite ou des noms alternatifs connus."""
    candidate_path = str(model_path)
    if os.path.exists(candidate_path):
        return joblib.load(candidate_path)

    models_dir = os.path.dirname(candidate_path) or 'models'
    base = os.path.basename(candidate_path)

    fallbacks = [
        base,
        'Random_Forest.pkl',
        'random_forest.pkl',
        'random_forest.joblib',
        'RandomForest.pkl',
    ]
    for name in fallbacks:
        path = os.path.join(models_dir, name)
        if os.path.exists(path):
            return joblib.load(path)

    raise FileNotFoundError(
        f"Aucun modèle trouvé. Chemin demandé: {candidate_path}. Dossier vérifié: {models_dir}"
    )


def train_and_evaluate_all(models, X_train, y_train, X_test, y_test, output_dir='reports/figures'):
    """Entraîne et évalue tous les modèles fournis."""
    results = {}
    for name, model in models.items():
        print(f"\n--- {name} ---")
        results[name] = evaluate_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            model_name=name,
            save_roc=True,
            output_dir=output_dir
        )
    return results


def build_results_dataframe(results):
    """Construit le DataFrame de comparaison des modèles."""
    return pd.DataFrame(results).T


def select_best_model(results_df, metric='ROC-AUC'):
    """Retourne le nom du meilleur modèle selon une métrique."""
    return results_df[metric].idxmax()


def train_and_save_best_model(best_name, X_train, y_train, models_dir='models'):
    """Entraîne et sauvegarde le meilleur modèle."""
    best_model = build_models()[best_name]
    best_model.fit(X_train, y_train)

    os.makedirs(models_dir, exist_ok=True)
    model_path = f"{models_dir}/{best_name.replace(' ', '_')}.pkl"
    joblib.dump(best_model, model_path)

    # Store metadata in JSON to avoid confusion with serialized estimators.
    info_path = f'{models_dir}/best_model_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump({'name': best_name, 'path': model_path}, f, ensure_ascii=False, indent=2)
    return best_model, model_path


def run_shap_reports(best_model, X_train, X_test, feature_names, figures_dir='reports/figures'):
    """Génère les sorties SHAP pour le meilleur modèle."""
    try:
        from .shap_explanations import generate_shap_summary, plot_waterfall
    except ImportError:
        from shap_explanations import generate_shap_summary, plot_waterfall

    os.makedirs(figures_dir, exist_ok=True)
    summary_result = generate_shap_summary(
        best_model,
        X_train,
        feature_names,
        output_path=f'{figures_dir}/shap_summary.png'
    )
    waterfall_result = plot_waterfall(
        best_model,
        X_instance=X_test[:1],
        feature_names=feature_names,
        X_background=X_train,
        output_path=f'{figures_dir}/shap_waterfall_sample.png'
    )
    return {'summary': summary_result, 'waterfall': waterfall_result}

def build_models():
    """Instancie les trois modèles sélectionnés."""
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }

    if LGBMClassifier is not None:
        models['LightGBM'] = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1
        )

    if CatBoostClassifier is not None:
        models['CatBoost'] = CatBoostClassifier(
            iterations=200,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            verbose=0
        )

    return models

def main():
    print("=" * 55)
    print("   Pipeline d'entraînement – Appendicite pédiatrique")
    print("=" * 55)

    # 1. Chargement et prétraitement
    print("\n[1/5] Chargement et prétraitement des données...")
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess(
        FILEPATH,
        target_col=TARGET_COL,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    feature_names = get_feature_names(preprocessor)
    print(f"      Features après transformation : {len(feature_names)}")

    # 2. Entraînement et évaluation
    print("\n[2/5] Entraînement et évaluation des modèles...")
    models = build_models()
    results = train_and_evaluate_all(
        models,
        X_train,
        y_train,
        X_test,
        y_test,
        output_dir='reports/figures'
    )

    # 3. Comparaison
    print("\n[3/5] Comparaison des modèles...")
    df_results = build_results_dataframe(results)
    print("\n" + df_results.round(4).to_string())

    os.makedirs('reports', exist_ok=True)
    df_results.to_csv('reports/model_comparison.csv')
    print("      Tableau sauvegardé → reports/model_comparison.csv")

    # 4. Sélection du meilleur modèle
    best_name = select_best_model(df_results, metric='ROC-AUC')
    print(f"\n[4/5] Meilleur modèle : {best_name}")
    print(f"      ROC-AUC = {df_results.loc[best_name, 'ROC-AUC']:.4f}")
    print(f"      F1-score = {df_results.loc[best_name, 'F1-score']:.4f}")

    best_model, model_path = train_and_save_best_model(
        best_name,
        X_train,
        y_train,
        models_dir='models'
    )
    print(f"      Modèle sauvegardé → {model_path}")

    # 5. SHAP
    print("\n[5/5] Génération des explications SHAP...")
    run_shap_reports(
        best_model,
        X_train,
        X_test,
        feature_names,
        figures_dir='reports/figures'
    )

    print("\n" + "=" * 55)
    print("  Entraînement terminé avec succès !")
    print("=" * 55)

if __name__ == '__main__':
    main()