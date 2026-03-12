"""
app/app.py
==========
Application FastAPI — aide au diagnostic pédiatrique de l'appendicite.

Fournit une interface web pour saisir les 10 paramètres cliniques d'un enfant
et obtenir une probabilité d'appendicite avec explication SHAP.

Features du modèle (10) :
  Catégorielles (0/1) : Lower_Right_Abd_Pain, Migratory_Pain, Nausea,
                        Ipsilateral_Rebound_Tenderness
  Numériques           : Body_Temperature, WBC_Count, CRP,
                         Neutrophil_Percentage, Appendix_Diameter, Age

Pour lancer :
  python app.py [--host 0.0.0.0] [--port 8000] [--reload]
  ou : uvicorn app:app --reload
"""

from __future__ import annotations

import sys
import pathlib

import joblib
import pandas as pd

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------------------------------
# sys.path — fonctionne quel que soit le répertoire de lancement
# ---------------------------------------------------------------------------
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.train_model import load_model
from src.evaluate_model import (
    predict_proba_safe,
    compute_shap_values,
    make_shap_waterfall_b64,
)

# ---------------------------------------------------------------------------
# Application FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PediAppendix",
    description="Aide au diagnostic pédiatrique de l'appendicite — Random Forest + SHAP",
    version="2.0.0",
)

_APP_DIR = pathlib.Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(_APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(_APP_DIR / "templates"))

# ---------------------------------------------------------------------------
# Chargement du modèle au démarrage (singleton)
# ---------------------------------------------------------------------------
_MODEL_DIR = _ROOT / "models"
_DATA_DIR = _ROOT / "data" / "processed"

_model = None
_feature_cols: list[str] = []
_defaults: dict[str, float] = {}

# Colonnes binaires du formulaire (encodées yes→1 / no→0)
_BINARY_COLS = {
    "lower_right_abd_pain",
    "migratory_pain",
    "nausea",
    "ipsilateral_rebound_tenderness",
}


def _load_resources() -> None:
    """Charge le modèle RF et les médianes du test set comme valeurs par défaut."""
    global _model, _feature_cols, _defaults
    _model = load_model(_MODEL_DIR / "random_forest.joblib")
    processed = joblib.load(_DATA_DIR / "processed_data.joblib")
    _feature_cols = processed["feature_cols"]
    _defaults = processed["X_test"].median().to_dict()


_load_resources()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_input_row(form_data: dict) -> pd.DataFrame:
    """
    Construit un DataFrame d'une ligne à partir des données brutes du formulaire.

    - Les champs numériques sont castés en float.
    - Les champs binaires yes/no sont encodés en 1/0.
    - Les valeurs manquantes sont remplacées par la médiane du test set.

    Retourne un DataFrame dont les colonnes correspondent exactement à _feature_cols.
    """
    row: dict[str, float] = dict(_defaults)

    # Champs numériques
    for key in (
        "body_temperature",
        "wbc_count",
        "crp",
        "neutrophil_percentage",
        "appendix_diameter",
        "age",
    ):
        # col = key.replace("_", "_").title().replace("_", "_")
        # Conversion key → nom de colonne (Body_Temperature, WBC_Count, …)
        col_name = {
            "body_temperature": "Body_Temperature",
            "wbc_count": "WBC_Count",
            "crp": "CRP",
            "neutrophil_percentage": "Neutrophil_Percentage",
            "appendix_diameter": "Appendix_Diameter",
            "age": "Age",
        }[key]
        try:
            row[col_name] = float(form_data[key])
        except (ValueError, KeyError):
            pass  # garde la médiane par défaut

    # Champs binaires yes/no → 1/0
    binary_map = {
        "lower_right_abd_pain": "Lower_Right_Abd_Pain",
        "migratory_pain": "Migratory_Pain",
        "nausea": "Nausea",
        "ipsilateral_rebound_tenderness": "Ipsilateral_Rebound_Tenderness",
    }
    for form_key, col_name in binary_map.items():
        val = str(form_data.get(form_key, "no")).strip().lower()
        row[col_name] = 1.0 if val == "yes" else 0.0

    return pd.DataFrame([{col: row[col] for col in _feature_cols}])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request) -> HTMLResponse:
    """Affiche le formulaire de saisie clinique."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "defaults": _defaults,
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    age: float = Form(...),
    body_temperature: float = Form(...),
    wbc_count: float = Form(...),
    crp: float = Form(...),
    neutrophil_percentage: float = Form(...),
    appendix_diameter: float = Form(...),
    lower_right_abd_pain: str = Form(default="no"),
    migratory_pain: str = Form(default="no"),
    nausea: str = Form(default="no"),
    ipsilateral_rebound_tenderness: str = Form(default="no"),
) -> HTMLResponse:
    """
    Route de prédiction.

    Reçoit les 10 champs du formulaire, construit le vecteur feature,
    calcule la probabilité d'appendicite, génère le graphique SHAP
    et renvoie la page résultat.
    """
    form_data = {
        "age": age,
        "body_temperature": body_temperature,
        "wbc_count": wbc_count,
        "crp": crp,
        "neutrophil_percentage": neutrophil_percentage,
        "appendix_diameter": appendix_diameter,
        "lower_right_abd_pain": lower_right_abd_pain,
        "migratory_pain": migratory_pain,
        "nausea": nausea,
        "ipsilateral_rebound_tenderness": ipsilateral_rebound_tenderness,
    }

    X = _build_input_row(form_data)
    prob = predict_proba_safe(_model, X)

    decision = "appendicite" if prob >= 0.5 else "pas d'appendicite"
    risk_class = "danger" if prob >= 0.5 else "success"

    # Graphique SHAP (non-bloquant en cas d'erreur)
    shap_b64: str | None = None
    try:
        sv, base_val = compute_shap_values(_model, X)
        shap_b64 = make_shap_waterfall_b64(sv, base_val, X)
    except Exception as exc:
        print(f"SHAP error (non-fatal): {exc}")

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "prob": f"{prob * 100:.1f}",
            "decision": decision,
            "risk_class": risk_class,
            "shap_b64": shap_b64,
            "form": form_data,
        },
    )


# ---------------------------------------------------------------------------
# Lancement direct avec arguments CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(
        description="PediAppendix — Aide au diagnostic pédiatrique de l'appendicite"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host (défaut: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (défaut: 8000)")
    parser.add_argument("--reload", action="store_true", help="Hot-reload")
    parser.add_argument("--log-level", default="info", help="Log level")
    args = parser.parse_args()

    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
