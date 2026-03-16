# src/train_model.py
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from data_processing import load_and_preprocess, get_feature_names
from evaluate_model import evaluate_model
from shap_explanations import generate_shap_summary, plot_waterfall

# Configuration
FILEPATH     = 'data/processed/data_finale.xlsx'
TARGET_COL   = 'Diagnosis'
TEST_SIZE    = 0.2
RANDOM_STATE = 42

def load_model(path):
    """Charge un modèle sauvegardé au format .joblib ou .pkl."""
    return joblib.load(path)

def build_models():
    """Instancie les trois modèles sélectionnés."""
    return {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=200,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            verbose=0
        ),
    }

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
    results = {}

    for name, model in models.items():
        print(f"\n--- {name} ---")
        results[name] = evaluate_model(
            model, X_train, y_train, X_test, y_test,
            model_name=name,
            save_roc=True,
            output_dir='reports/figures'
        )

    # 3. Comparaison
    print("\n[3/5] Comparaison des modèles...")
    df_results = pd.DataFrame(results).T
    print("\n" + df_results.round(4).to_string())

    os.makedirs('reports', exist_ok=True)
    df_results.to_csv('reports/model_comparison.csv')
    print("      Tableau sauvegardé → reports/model_comparison.csv")

    # 4. Sélection du meilleur modèle
    best_name = df_results['ROC-AUC'].idxmax()
    print(f"\n[4/5] Meilleur modèle : {best_name}")
    print(f"      ROC-AUC = {df_results.loc[best_name, 'ROC-AUC']:.4f}")
    print(f"      F1-score = {df_results.loc[best_name, 'F1-score']:.4f}")

    best_model = build_models()[best_name]
    best_model.fit(X_train, y_train)

    model_path = f"models/{best_name.replace(' ', '_')}.pkl"
    joblib.dump(best_model, model_path)
    print(f"      Modèle sauvegardé → {model_path}")
    joblib.dump({'name': best_name, 'path': model_path}, 'models/best_model_info.pkl')

    # 5. SHAP
    print("\n[5/5] Génération des explications SHAP...")
    os.makedirs('reports/figures', exist_ok=True)

    generate_shap_summary(
        best_model, X_train, feature_names,
        output_path='reports/figures/shap_summary.png'
    )

    plot_waterfall(
        best_model,
        X_instance=X_test[:1],
        feature_names=feature_names,
        X_background=X_train,
        output_path='reports/figures/shap_waterfall_sample.png'
    )

    print("\n" + "=" * 55)
    print("  Entraînement terminé avec succès !")
    print("=" * 55)

if __name__ == '__main__':
    main()