# src/evaluate_model.py
import numpy as np
import io
import base64
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    classification_report, RocCurveDisplay
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import os


def predict_proba_safe(model, X):
    """
    Retourne la probabilité positive pour la première ligne de `X`.

    Supporte différents types d'estimateurs : utilise `predict_proba` si disponible,
    fallback sur `decision_function` (sigmoïdation) ou `predict` pour compatibilité.
    Justification : rend l'inférence robuste face aux variations d'API des modèles.
    """
    if hasattr(model, 'predict_proba'):
        return float(model.predict_proba(X)[0, 1])

    if hasattr(model, 'decision_function'):
        score = float(np.ravel(model.decision_function(X))[0])
        return float(1.0 / (1.0 + np.exp(-score)))

    pred = float(np.ravel(model.predict(X))[0])
    return max(0.0, min(1.0, pred))


def compute_shap_values(model, X, X_background=None):
    """
    Calcule les valeurs SHAP pour une (ou plusieurs) instance(s) `X`.

    Charge dynamiquement les utilitaires SHAP et s'assure que SHAP est disponible.
    Retourne un tuple `(values_array, base_value)` utilisable pour visualisations.
    Justification : centralise la logique SHAP et gère l'absence de dépendance proprement.
    """
    try:
        from .shap_explanations import shap_ready, compute_shap_values as compute_raw_shap
    except ImportError:
        from shap_explanations import shap_ready, compute_shap_values as compute_raw_shap

    if not shap_ready():
        raise RuntimeError("SHAP indisponible")

    background = X_background if X_background is not None else X
    explainer, values = compute_raw_shap(model, X, background)
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1]
    return np.array(values[0]), float(base_value)


def make_shap_waterfall_b64(shap_values, base_value, X):
    """
    Génère un graphique horizontal compact des principales contributions SHAP encodé en base64.

    Sélectionne les 10 features absolues les plus importantes et produit un PNG encodé.
    Permet d'inclure rapidement une image SHAP dans des templates HTML sans fichiers temporaires.
    """
    values = np.array(shap_values)
    features = np.array(X.iloc[0] if hasattr(X, 'iloc') else np.ravel(X))
    labels = list(X.columns) if hasattr(X, 'columns') else [f"f{i}" for i in range(len(values))]

    order = np.argsort(np.abs(values))[::-1][:10]
    vals = values[order]
    lbls = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#d62728' if v > 0 else '#1f77b4' for v in vals]
    ax.barh(range(len(vals)), vals, color=colors)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(lbls)
    ax.invert_yaxis()
    ax.set_title(f"Contributions SHAP (base={base_value:.3f})")
    ax.set_xlabel("Contribution")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def fit_estimator(model, X_train, y_train):
    """
    Entraîne l'estimateur sur les données fournies et retourne l'objet entraîné.

    Simple wrapper pour rendre explicite l'étape d'entraînement dans le pipeline
    et faciliter le débogage / point d'interruption lors des runs.
    """
    model.fit(X_train, y_train)
    return model


def predict_labels_and_proba(model, X_test):
    """
    Retourne à la fois les labels prédits et les probabilités de la classe positive.

    Sépare prédiction de classes et probabilités pour calculer différentes métriques.
    Assure compatibilité avec la plupart des API sklearn qui exposent `predict_proba`.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return y_pred, y_proba


def compute_classification_metrics(y_test, y_pred, y_proba):
    """
    Calcule métriques de classification usuelles et retourne un dict.

    Inclut ROC-AUC, accuracy, precision, recall et F1-score; protège contre
    divisions par zéro via `zero_division=0` pour stabilité sur petits jeux.
    """
    return {
        'ROC-AUC': roc_auc_score(y_test, y_proba),
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-score': f1_score(y_test, y_pred, zero_division=0),
    }


def compute_cv_auc(model, X_train, y_train, n_splits=5, random_state=42):
    """
    Effectue une validation croisée stratifiée et retourne la moyenne et l'écart-type ROC-AUC.

    Utilise `StratifiedKFold` pour préserver la distribution de la classe et donner
    une estimation plus stable des performances du modèle.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    return cv_scores.mean(), cv_scores.std()


def maybe_save_roc_curve(y_test, y_proba, model_name, output_dir='reports/figures', enabled=False):
    """
    Sauvegarde la courbe ROC dans `output_dir` si `enabled=True`.

    Permet d'activer la génération d'artefacts visuels uniquement quand nécessaire
    (ex. lors d'expériences), évitant coûts CPU/IO inutiles en production.
    """
    if not enabled:
        return None

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name=model_name)
    ax.set_title(f"ROC Curve - {model_name}")
    path = os.path.join(output_dir, f"roc_{model_name.replace(' ', '_')}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def render_evaluation_report(metrics, y_test, y_pred, model_name='Model'):
    """
    Affiche un rapport d'évaluation lisible dans la console.

    Imprime métriques formatées, matrice de confusion et classification report.
    Utile pour runs CLI et debugging rapide des performances.
    """
    print(f"\n{'─'*45}")
    print(f"  {model_name}")
    print(f"{'─'*45}")
    for k, v in metrics.items():
        print(f"  {k:<22}: {v:.4f}")
    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, y_pred))
    print("\n" + classification_report(y_test, y_pred, zero_division=0))

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name='Model', save_roc=False, output_dir='reports/figures'):
    """
    Orchestrateur d'évaluation : entraine, évalue sur test et exécute CV.

    Retourne un dictionnaire de métriques consolidées et peut sauvegarder la ROC.
    Conçu pour être utilisé dans des comparaisons de modèles automatisées.
    """
    fit_estimator(model, X_train, y_train)
    y_pred, y_proba = predict_labels_and_proba(model, X_test)

    metrics = compute_classification_metrics(y_test, y_pred, y_proba)
    cv_mean, cv_std = compute_cv_auc(model, X_train, y_train)
    metrics['CV ROC-AUC mean'] = cv_mean
    metrics['CV ROC-AUC std'] = cv_std

    render_evaluation_report(metrics, y_test, y_pred, model_name=model_name)

    roc_path = maybe_save_roc_curve(
        y_test,
        y_proba,
        model_name=model_name,
        output_dir=output_dir,
        enabled=save_roc
    )
    if roc_path:
        print(f"ROC curve saved → {roc_path}")

    return metrics