"""
src/evaluate_model.py
Paradigme fonctionnel : une fonction = une tâche précise et testable.
Évaluation des modèles et génération des visualisations SHAP.
"""

from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — pas de display requis
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.train_model import load_model


# ---------------------------------------------------------------------------
# 1. Métriques — une fonction par calcul
# ---------------------------------------------------------------------------

def predict_proba_safe(model: object, X: pd.DataFrame) -> np.ndarray:
    """
    Retourne les probabilités de la classe positive (1).
    Gère les modèles avec/sans méthode predict_proba.

    Returns
    -------
    Array 1D de probabilités.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # Normalisation sigmoidale
        return 1 / (1 + np.exp(-scores))
    raise ValueError(f"Le modèle {type(model).__name__} ne supporte ni predict_proba ni decision_function.")


def compute_metrics(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Calcule toutes les métriques d'évaluation pour un modèle.

    Returns
    -------
    Dictionnaire : accuracy, precision, recall, f1, roc_auc.
    """
    y_proba = predict_proba_safe(model, X_test)
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
    }


def compare_models(
    models: dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Compare plusieurs modèles sur toutes les métriques.

    Parameters
    ----------
    models : dictionnaire {nom: modèle}

    Returns
    -------
    DataFrame trié par roc_auc décroissant.
    """
    rows = []
    for name, model in models.items():
        metrics = compute_metrics(model, X_test, y_test)
        rows.append({"model": name, **metrics})
    df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. Visualisations — une fonction par graphique
# ---------------------------------------------------------------------------

def plot_roc_curves(
    models: dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: str | pathlib.Path | None = None,
) -> plt.Figure:
    """
    Trace les courbes ROC de tous les modèles sur un même graphique.

    Returns
    -------
    Figure matplotlib.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models.items():
        y_proba = predict_proba_safe(model, X_test)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aléatoire")
    ax.set_xlabel("Taux de Faux Positifs")
    ax.set_ylabel("Taux de Vrais Positifs")
    ax.set_title("Courbes ROC — comparaison des modèles")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Modèle",
    save_path: str | pathlib.Path | None = None,
) -> plt.Figure:
    """
    Trace la matrice de confusion normalisée.

    Returns
    -------
    Figure matplotlib.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["No App.", "Appendicite"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Matrice de confusion — {model_name}")
    plt.tight_layout()

    if save_path:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 3. SHAP — une fonction par type de visualisation
# ---------------------------------------------------------------------------

def compute_shap_values(
    model: object,
    X: pd.DataFrame,
    max_display: int = 200,
) -> tuple:
    """
    Calcule les valeurs SHAP pour un modèle tree-based ou générique.

    Utilise TreeExplainer si disponible (RF, LightGBM, CatBoost),
    sinon KernelExplainer sur un sous-échantillon.

    Returns
    -------
    (explainer, shap_values) — shap_values est un array numpy.
    """
    import shap
    shap.initjs()

    # TreeExplainer pour les modèles basés sur des arbres
    try:
        inner = model.named_steps["svc"] if hasattr(model, "named_steps") else model
        explainer = shap.TreeExplainer(inner)
        sv = explainer.shap_values(X)
        # SHAP 0.4+ renvoie selon la version:
        # - list de 2 arrays (n, f)  → binary RF
        # - numpy object array shape (2,) contenant deux (n, f)
        # - numpy 3D array (n, f, 2) or (2, n, f)
        # Dans tous les cas, on sélectionne la classe positive (1)
        if isinstance(sv, list) and len(sv) == 2:
            sv = sv[1]
        elif isinstance(sv, np.ndarray):
            if sv.ndim == 1 and sv.dtype == object and len(sv) == 2:
                sv = sv[1]          # object array: [sv_class0, sv_class1]
            elif sv.ndim == 3:
                if sv.shape[-1] == 2:   # (n, f, 2)
                    sv = sv[:, :, 1]
                elif sv.shape[0] == 2:  # (2, n, f)
                    sv = sv[1]
        return explainer, sv
    except Exception:
        pass

    # KernelExplainer comme fallback (lent, sous-échantillon)
    sample = shap.sample(X, min(max_display, len(X)))
    explainer = shap.KernelExplainer(
        lambda x: _proba_wrapper(model, x, X.columns), sample
    )
    sv = explainer.shap_values(sample, nsamples=100)
    return explainer, sv


def _proba_wrapper(model, x_array, columns):
    """Wrapper interne pour KernelExplainer — convertit numpy → DataFrame."""
    X_df = pd.DataFrame(x_array, columns=columns)
    return predict_proba_safe(model, X_df)


def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    max_display: int = 20,
    save_path: str | pathlib.Path | None = None,
) -> plt.Figure:
    """
    Trace le SHAP summary plot (beeswarm).

    Returns
    -------
    Figure matplotlib.
    """
    import shap

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    plt.title("SHAP — Impact des variables sur le diagnostic (appendicite=1)")
    plt.tight_layout()

    if save_path:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return plt.gcf()


def plot_shap_waterfall(
    explainer: object,
    shap_values: np.ndarray,
    X: pd.DataFrame,
    sample_idx: int = 0,
    save_path: str | pathlib.Path | None = None,
) -> plt.Figure:
    """
    Trace un waterfall plot SHAP pour un patient individuel.

    Parameters
    ----------
    sample_idx : index de la ligne dans X à expliquer

    Returns
    -------
    Figure matplotlib.
    """
    import shap

    # Sélectionner la valeur de base pour la classe positive
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)) and len(ev) == 2:
        base_val = float(ev[1])
    else:
        base_val = float(ev)

    # Extraire les valeurs SHAP pour le sample demandé
    sv_sample = shap_values[sample_idx]

    # Si shap_values est encore un array 2D (n, 2) ou (2, n), extraire classe positive
    if isinstance(sv_sample, np.ndarray) and sv_sample.ndim == 1 and len(sv_sample) == 2:
        # Object array edge-case (rare)
        sv_sample = sv_sample.ravel()
    elif isinstance(sv_sample, np.ndarray) and sv_sample.ndim == 2:
        # (n_features, 2) or similar — take column 1
        sv_sample = sv_sample[:, 1]

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=sv_sample,
            base_values=base_val,
            data=X.iloc[sample_idx].values,
            feature_names=X.columns.tolist(),
        ),
        show=False,
    )
    plt.title(f"SHAP Waterfall — Patient #{sample_idx}")
    plt.tight_layout()

    if save_path:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return plt.gcf()


def plot_shap_dependence(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature: str,
    interaction_feature: str = "auto",
    save_path: str | pathlib.Path | None = None,
) -> plt.Figure:
    """
    Trace un dependence plot SHAP pour une variable clinique.

    Returns
    -------
    Figure matplotlib.
    """
    import shap

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(
        feature, shap_values, X,
        interaction_index=interaction_feature,
        ax=ax, show=False,
    )
    plt.title(f"SHAP Dependence — {feature}")
    plt.tight_layout()

    if save_path:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 4. Pipeline d'évaluation complet
# ---------------------------------------------------------------------------

def run_evaluation_pipeline(
    model_dir: str | pathlib.Path = "models",
    output_dir: str | pathlib.Path = "app/static/plots",
) -> pd.DataFrame:
    """
    Charge tous les modèles sauvegardés, compare leurs métriques,
    génère les courbes ROC et les plots SHAP du meilleur modèle.

    Returns
    -------
    DataFrame de comparaison des métriques.
    """
    import joblib

    model_dir = pathlib.Path(model_dir)
    output_dir = pathlib.Path(output_dir)

    # Chargement des données de test
    test_data = joblib.load(model_dir / "test_data.joblib")
    X_test, y_test = test_data["X_test"], test_data["y_test"]

    # Chargement de tous les modèles disponibles
    models = {}
    for p in model_dir.glob("*.joblib"):
        if p.stem == "test_data":
            continue
        models[p.stem] = load_model(p)

    if not models:
        raise FileNotFoundError(f"Aucun modèle trouvé dans {model_dir}")

    # Comparaison
    comparison = compare_models(models, X_test, y_test)
    print(comparison.to_string(index=False))

    # Courbes ROC
    plot_roc_curves(models, X_test, y_test, save_path=output_dir / "roc_curves.png")

    # SHAP sur le meilleur modèle
    best_name = comparison.iloc[0]["model"]
    best_model = models[best_name]
    print(f"\nMeilleur modèle : {best_name} (AUC={comparison.iloc[0]['roc_auc']:.4f})")

    explainer, sv = compute_shap_values(best_model, X_test)
    plot_shap_summary(sv, X_test, save_path=output_dir / "shap_summary.png")
    plot_shap_waterfall(explainer, sv, X_test, sample_idx=0,
                        save_path=output_dir / "shap_waterfall_0.png")

    return comparison


if __name__ == "__main__":
    run_evaluation_pipeline()
