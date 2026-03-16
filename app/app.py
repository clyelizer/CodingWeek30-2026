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

Design : templates premium (landing_page, auth avec flip 3D, diagnosis_console).
Auth   : session cookie signé HMAC — admin/admin123 (sans base de données).
"""

from __future__ import annotations

import sys
import pathlib
import os
import time
import base64
import hashlib
import hmac
import json

import joblib
import pandas as pd

from fastapi import FastAPI, Request, Form, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------------------------------
# sys.path — fonctionne quel que soit le répertoire de lancement
# ---------------------------------------------------------------------------
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.train_model import load_model
from src.evaluate_model import predict_proba_safe, compute_shap_values, make_shap_waterfall_b64

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
_MODEL_DIR  = _ROOT / "models"
_DATA_DIR   = _ROOT / "data" / "processed"

_model         = None
_feature_cols: list[str] = []
_defaults: dict[str, float] = {}

# Colonnes binaires du formulaire (encodées yes→1 / no→0)
_BINARY_COLS = {
    "lower_right_abd_pain",
    "migratory_pain",
    "nausea",
    "ipsilateral_rebound_tenderness",
}

# ---------------------------------------------------------------------------
# Auth : session simplifiée (HMAC, sans base de données)
# ---------------------------------------------------------------------------
_SECRET_KEY = os.environ.get("PEDIA_SECRET", "dev-secret-please-change")
_SESSION_COOKIE = "pedi_session"
_SESSION_DURATION = 4 * 60 * 60  # 4 heures

# Utilisateur administrateur codé en dur (usage local / démo)
_ADMIN_USER = "admin"
_ADMIN_PASS = "admin123"


def _make_session_token(username: str) -> str:
    """Génère un token de session signé HMAC, encodé URL-safe base64."""
    expires = int(time.time()) + _SESSION_DURATION
    payload = f"{username}|{expires}"
    signature = hmac.new(_SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(f"{payload}|{signature}".encode()).decode()


def _verify_session_token(token: str) -> str | None:
    """Vérifie un token et retourne le username ou None."""
    try:
        raw = base64.urlsafe_b64decode(token.encode()).decode()
        username, expires, signature = raw.split("|")
        expected = hmac.new(_SECRET_KEY.encode(), f"{username}|{expires}".encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return None
        if int(expires) < int(time.time()):
            return None
        return username
    except Exception:
        return None


def _get_current_user(request: Request) -> str | None:
    token = request.cookies.get(_SESSION_COOKIE)
    if not token:
        return None
    return _verify_session_token(token)


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
    for key in ("body_temperature", "wbc_count", "crp",
                "neutrophil_percentage", "appendix_diameter", "age"):
        col = key.replace("_", "_").title().replace("_", "_")
        # Conversion key → nom de colonne (Body_Temperature, WBC_Count, …)
        col_name = {
            "body_temperature":     "Body_Temperature",
            "wbc_count":            "WBC_Count",
            "crp":                  "CRP",
            "neutrophil_percentage":"Neutrophil_Percentage",
            "appendix_diameter":    "Appendix_Diameter",
            "age":                  "Age",
        }[key]
        try:
            row[col_name] = float(form_data[key])
        except (ValueError, KeyError):
            pass  # garde la médiane par défaut

    # Champs binaires yes/no → 1/0
    binary_map = {
        "lower_right_abd_pain":           "Lower_Right_Abd_Pain",
        "migratory_pain":                 "Migratory_Pain",
        "nausea":                         "Nausea",
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
    """Page d'accueil (landing page premium)."""
    return templates.TemplateResponse("landing_page.html", {"request": request})


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> HTMLResponse:
    """Page de connexion — animation flip login / créer un compte."""
    if _get_current_user(request):
        return RedirectResponse("/form", status_code=303)
    return templates.TemplateResponse("auth.html", {"request": request})


@app.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    """Vérifie admin/admin123 et pose un cookie de session signé."""
    if username == _ADMIN_USER and password == _ADMIN_PASS:
        token = _make_session_token(username)
        response = RedirectResponse(url="/form", status_code=303)
        response.set_cookie(_SESSION_COOKIE, token, httponly=True, samesite="lax")
        return response
    return templates.TemplateResponse(
        "auth.html",
        {"request": request, "error": "Identifiant ou mot de passe incorrect."},
        status_code=401,
    )


@app.post("/register")
async def register(request: Request):
    """Stub création de compte — renvoie vers auth avec message."""
    return templates.TemplateResponse(
        "auth.html",
        {"request": request, "error": "La création de compte est désactivée en mode démo. Utilisez admin / admin123."},
        status_code=200,
    )


@app.get("/logout")
async def logout() -> RedirectResponse:
    """Déconnecte l'utilisateur."""
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(_SESSION_COOKIE)
    return response


@app.get("/form", response_class=HTMLResponse)
async def read_form(request: Request) -> HTMLResponse:
    """Affiche la console de diagnostic clinique."""
    user = _get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse("diagnosis_console.html", {
        "request":  request,
        "defaults": _defaults,
        "user":     user,
    })


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request:                        Request,
    age:                            float = Form(...),
    body_temperature:               float = Form(...),
    wbc_count:                      float = Form(...),
    crp:                            float = Form(...),
    neutrophil_percentage:          float = Form(...),
    appendix_diameter:              float = Form(...),
    lower_right_abd_pain:           str   = Form(default="no"),
    migratory_pain:                 str   = Form(default="no"),
    nausea:                         str   = Form(default="no"),
    ipsilateral_rebound_tenderness: str   = Form(default="no"),
) -> HTMLResponse:
    """
    Route de prédiction HTML.
    Rend la console de diagnostic avec les résultats inline.
    """
    user = _get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    form_data = {
        "age":                            age,
        "body_temperature":               body_temperature,
        "wbc_count":                      wbc_count,
        "crp":                            crp,
        "neutrophil_percentage":          neutrophil_percentage,
        "appendix_diameter":              appendix_diameter,
        "lower_right_abd_pain":           lower_right_abd_pain,
        "migratory_pain":                 migratory_pain,
        "nausea":                         nausea,
        "ipsilateral_rebound_tenderness": ipsilateral_rebound_tenderness,
    }

    X    = _build_input_row(form_data)
    
    # Vérifier que le modèle est chargé
    if _model is None:
        raise RuntimeError("Le modèle n'a pas pu être chargé")
    
    prob = predict_proba_safe(_model, X)

    decision   = "appendicite" if prob >= 0.5 else "pas d'appendicite"
    risk_class = "danger"      if prob >= 0.5 else "success"

    # Graphique SHAP (non-bloquant en cas d'erreur)
    shap_b64: str | None = None
    try:
        sv, base_val = compute_shap_values(_model, X)
        shap_b64 = make_shap_waterfall_b64(sv, base_val, X)
    except Exception as exc:
        print(f"SHAP error (non-fatal): {exc}")

    return templates.TemplateResponse("diagnosis_console.html", {
        "request":    request,
        "user":       user,
        "prob":       f"{prob * 100:.1f}",
        "decision":   decision,
        "risk_class": risk_class,
        "shap_b64":   shap_b64,
        "form":       form_data,
        "defaults":   _defaults,
    })


# ---------------------------------------------------------------------------
# API JSON (pour preview temps réel dans diagnosis_console.html)
# ---------------------------------------------------------------------------

@app.post("/api/predict")
async def api_predict(request: Request) -> dict:
    """API JSON pour prédiction temps réel — utilisée par le frontend AJAX."""
    if not _get_current_user(request):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Non autorisé")

    form_values = await request.form()
    form_data: dict[str, str | float] = {k: v for k, v in form_values.items()}

    X    = _build_input_row(form_data)
    prob = predict_proba_safe(_model, X)

    decision   = "appendicite" if prob >= 0.5 else "pas d'appendicite"
    risk_class = "danger"      if prob >= 0.5 else "success"

    shap_b64: str | None = None
    try:
        sv, base_val = compute_shap_values(_model, X)
        shap_b64 = make_shap_waterfall_b64(sv, base_val, X)
    except Exception as exc:
        print(f"SHAP error (non-fatal): {exc}")

    return {
        "prob":       f"{prob * 100:.1f}",
        "decision":   decision,
        "risk_class": risk_class,
        "shap_b64":   shap_b64,
        "form":       form_data,
    }


# ---------------------------------------------------------------------------
# Lancement direct avec arguments CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(
        description="PediAppendix — Aide au diagnostic pédiatrique de l'appendicite"
    )
    parser.add_argument("--host",      default="0.0.0.0",  help="Host (défaut: 0.0.0.0)")
    parser.add_argument("--port",      type=int, default=8000, help="Port (défaut: 8000)")
    parser.add_argument("--reload",    action="store_true", help="Hot-reload")
    parser.add_argument("--log-level", default="info",      help="Log level")
    args = parser.parse_args()

    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
