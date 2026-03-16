# src/shap_explanations.py
import numpy as np
import matplotlib.pyplot as plt
import os

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Attention : shap non installé. Installez-le avec : pip install shap")

def _get_explainer(model, X_background):
    """Choisit automatiquement le bon explainer selon le type de modèle."""
    class_name = type(model).__name__
    tree_models = ('RandomForestClassifier', 'GradientBoostingClassifier',
                   'ExtraTreesClassifier', 'DecisionTreeClassifier',
                   'LGBMClassifier', 'XGBClassifier', 'CatBoostClassifier')
    if class_name in tree_models:
        return shap.TreeExplainer(model)
    else:
        background = shap.sample(X_background, min(100, len(X_background)))
        return shap.KernelExplainer(model.predict_proba, background)

def _extract_shap_values(shap_values):
    """Extrait les valeurs pour la classe positive (index 1)."""
    if isinstance(shap_values, list):
        return shap_values[1]
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values

def generate_shap_summary(model, X_train, feature_names, output_path='reports/figures/shap_summary.png', max_samples=200):
    """Génère un summary plot SHAP (beeswarm + bar)."""
    if not SHAP_AVAILABLE:
        print("shap non disponible, summary plot ignoré.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n = min(max_samples, X_train.shape[0])
    X_sample = X_train[:n]

    explainer = _get_explainer(model, X_train)
    shap_values = explainer.shap_values(X_sample)
    sv = _extract_shap_values(shap_values)

    # Beeswarm plot
    plt.figure(figsize=(10,7))
    shap.summary_plot(sv, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Summary plot → {output_path}")

    # Bar plot
    bar_path = output_path.replace('.png', '_bar.png')
    plt.figure(figsize=(10,7))
    shap.summary_plot(sv, X_sample, feature_names=feature_names, plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Bar plot     → {bar_path}")

def get_shap_values(model, X_instance, X_background):
    """Retourne les valeurs SHAP pour une instance unique."""
    if not SHAP_AVAILABLE:
        return None
    explainer = _get_explainer(model, X_background)
    shap_values = explainer.shap_values(X_instance)
    return _extract_shap_values(shap_values)

def plot_waterfall(model, X_instance, feature_names, X_background, output_path='reports/figures/shap_waterfall.png'):
    """Génère un waterfall plot pour une instance unique."""
    if not SHAP_AVAILABLE:
        print("shap non disponible, waterfall ignoré.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    explainer = _get_explainer(model, X_background)
    shap_values = explainer.shap_values(X_instance)
    sv = _extract_shap_values(shap_values)

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1]

    exp = shap.Explanation(
        values=sv[0],
        base_values=base_value,
        data=X_instance[0],
        feature_names=feature_names
    )

    plt.figure()
    shap.waterfall_plot(exp, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Waterfall plot → {output_path}")