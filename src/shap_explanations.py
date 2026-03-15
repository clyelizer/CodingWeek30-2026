# src/shap_explanations.py
import numpy as np
import matplotlib.pyplot as plt
import os

shap = None
SHAP_AVAILABLE = None


def _load_shap():
    """Charge SHAP à la demande et met en cache le statut de disponibilité."""
    global shap, SHAP_AVAILABLE
    if SHAP_AVAILABLE is not None:
        return SHAP_AVAILABLE
    try:
        import shap as shap_module
        shap = shap_module
        SHAP_AVAILABLE = True
    except ImportError:
        SHAP_AVAILABLE = False
    return SHAP_AVAILABLE


def shap_ready():
    """Indique si SHAP est disponible."""
    return _load_shap()


def ensure_output_dir(path):
    """Crée le dossier parent d'un fichier de sortie."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

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


def compute_shap_values(model, X_sample, X_background):
    """Calcule les valeurs SHAP normalisées pour un échantillon."""
    explainer = _get_explainer(model, X_background)
    raw = explainer.shap_values(X_sample)
    values = _extract_shap_values(raw)
    return explainer, values


def build_waterfall_explanation(explainer, shap_values, X_instance, feature_names):
    """Construit un objet SHAP Explanation pour waterfall."""
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1]

    return shap.Explanation(
        values=shap_values[0],
        base_values=base_value,
        data=X_instance[0],
        feature_names=feature_names
    )


def save_shap_summary_plots(sv, X_sample, feature_names, output_path):
    """Sauvegarde summary plot (beeswarm) et bar plot."""
    ensure_output_dir(output_path)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    bar_path = output_path.replace('.png', '_bar.png')
    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, X_sample, feature_names=feature_names, plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path, bar_path


def save_waterfall_plot(explanation, output_path):
    """Sauvegarde un waterfall plot à partir d'une explanation SHAP."""
    ensure_output_dir(output_path)
    plt.figure()
    shap.waterfall_plot(explanation, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path

def generate_shap_summary(model, X_train, feature_names, output_path='reports/figures/shap_summary.png', max_samples=200):
    """Génère un summary plot SHAP (beeswarm + bar)."""
    if not shap_ready():
        msg = "shap non disponible, summary plot ignoré."
        print(msg)
        return {'status': 'skipped', 'reason': msg}

    n = min(max_samples, X_train.shape[0])
    X_sample = X_train[:n]

    _, sv = compute_shap_values(model, X_sample, X_train)
    summary_path, bar_path = save_shap_summary_plots(sv, X_sample, feature_names, output_path)

    print(f"Summary plot → {summary_path}")
    print(f"Bar plot     → {bar_path}")
    return {'status': 'ok', 'summary_path': summary_path, 'bar_path': bar_path}

def get_shap_values(model, X_instance, X_background):
    """Retourne les valeurs SHAP pour une instance unique."""
    if not shap_ready():
        return None
    _, values = compute_shap_values(model, X_instance, X_background)
    return values

def plot_waterfall(model, X_instance, feature_names, X_background, output_path='reports/figures/shap_waterfall.png'):
    """Génère un waterfall plot pour une instance unique."""
    if not shap_ready():
        msg = "shap non disponible, waterfall ignoré."
        print(msg)
        return {'status': 'skipped', 'reason': msg}

    explainer, sv = compute_shap_values(model, X_instance, X_background)
    exp = build_waterfall_explanation(explainer, sv, X_instance, feature_names)
    saved_path = save_waterfall_plot(exp, output_path)
    print(f"Waterfall plot → {saved_path}")
    return {'status': 'ok', 'waterfall_path': saved_path}