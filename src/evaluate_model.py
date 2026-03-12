"""
src/evaluate_model.py
=====================
Évaluation des modèles et génération des explications SHAP.

Paradigme fonctionnel strict :
  - Une fonction = une tâche précise et testable.
  - Pas d'état global mutable.
  - Import de shap réalisé en lazy (à l'intérieur des fonctions) pour éviter
    de bloquer le chargement du module à cause de la chaîne scipy → shap.

Fonctions :
  1. predict_proba_safe      → probabilité d'appendicite pour un patient
  2. compute_shap_values     → valeurs SHAP pour une observation
  3. make_shap_waterfall_b64 → graphique waterfall encodé en base64 (pour HTML)
  4. build_results_summary   → récapitulatif des métriques de tous les modèles
"""

from __future__ import annotations

import io
import base64
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# 1. Prédiction
# ---------------------------------------------------------------------------

def predict_proba_safe(pipeline: Pipeline, X: pd.DataFrame) -> float:
    """
    Retourne la probabilité d'appendicite (classe 1) pour une observation.

    Paramètres
    ----------
    pipeline : Pipeline sklearn fitted
    X        : DataFrame d'une seule ligne (les 10 features)

    Retourne
    --------
    float ∈ [0, 1]

    Assertion testée : la probabilité retournée est dans [0, 1].
    """
    proba = float(pipeline.predict_proba(X)[0, 1])
    assert 0.0 <= proba <= 1.0, f"Probabilité invalide : {proba}"
    return proba


# ---------------------------------------------------------------------------
# 2. Valeurs SHAP
# ---------------------------------------------------------------------------

def compute_shap_values(
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    """
    Calcule les valeurs SHAP pour la première ligne de X via TreeExplainer.

    Utilise SHAP TreeExplainer sur le clf extrait du pipeline.
    Le scaler est appliqué manuellement avant de passer au TreeExplainer
    (qui attend les données dans l'espace du clf, pas l'espace original).

    Paramètres
    ----------
    pipeline : Pipeline sklearn fitted (scaler + clf)
    X        : DataFrame avec au moins une ligne

    Retourne
    --------
    (shap_values, base_value) : tuple
      - shap_values : array 1D de longueur n_features
      - base_value  : float, la valeur de base de l'explainer

    Assertion testée : len(shap_values) == nombre de features de X.
    """
    import shap  # import lazy — évite scipy au chargement du module

    clf = pipeline.named_steps["clf"]
    scaler = pipeline.named_steps["scaler"]
    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
    )

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_scaled)

    # sv peut être une liste [classe0, classe1] ou un array 3D selon la version
    if isinstance(sv, list):
        sv_single = sv[1][0] if len(sv) == 2 else sv[0][0]
    elif sv.ndim == 3:
        sv_single = sv[0, :, 1]
    else:
        sv_single = sv[0]

    base_val = float(
        explainer.expected_value[1]
        if hasattr(explainer.expected_value, "__len__")
        else explainer.expected_value
    )

    assert len(sv_single) == X.shape[1], (
        f"SHAP : {len(sv_single)} valeurs pour {X.shape[1]} features"
    )
    return sv_single, base_val


# ---------------------------------------------------------------------------
# 3. Graphique SHAP waterfall encodé base64
# ---------------------------------------------------------------------------

def make_shap_waterfall_b64(
    shap_values: np.ndarray,
    base_value: float,
    X_row: pd.DataFrame,
) -> str | None:
    """
    Génère un graphique waterfall SHAP et le retourne encodé en base64 (PNG).

    Retourne None en cas d'erreur (non-fatal pour l'interface web).

    Assertion testée : la chaîne retournée est décodable en bytes non vides.
    """
    import shap  # import lazy

    try:
        fig, _ = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values,
                base_values=base_value,
                data=X_row.iloc[0].values,
                feature_names=list(X_row.columns),
            ),
            show=False,
        )
        plt.title("Explication SHAP — Facteurs influençant la prédiction")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close("all")

        assert len(b64) > 0, "Image SHAP vide"
        return b64

    except Exception as exc:
        plt.close("all")
        print(f"SHAP waterfall error (non-fatal): {exc}")
        return None


# ---------------------------------------------------------------------------
# 4. Récapitulatif des métriques
# ---------------------------------------------------------------------------

def build_results_summary(
    results: dict[str, dict[str, float]],
    best_name: str,
) -> pd.DataFrame:
    """
    Construit un DataFrame récapitulatif des métriques de tous les modèles.

    Paramètres
    ----------
    results   : {nom_modèle: {roc_auc, f1, accuracy}}
    best_name : nom du meilleur modèle (marqué dans le DataFrame)

    Retourne
    --------
    pd.DataFrame trié par roc_auc décroissant.

    Assertion testée : le DataFrame a autant de lignes que de modèles dans results.
    """
    rows = []
    for name, metrics in results.items():
        rows.append({
            "model":    name,
            "roc_auc":  round(metrics["roc_auc"], 4),
            "f1":       round(metrics["f1"], 4),
            "accuracy": round(metrics["accuracy"], 4),
            "best":     name == best_name,
        })
    df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    assert len(df) == len(results), "Nombre de lignes incorrect dans le récapitulatif"
    return df
