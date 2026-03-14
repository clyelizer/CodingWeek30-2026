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
import sqlite3
import json
import os
import time
import base64
import hashlib
import hmac
from datetime import datetime

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
_DB_PATH = _ROOT / "data" / "patient_history.db"

# Secret for session signing (change in production via env var)
_SECRET_KEY = os.environ.get("PEDIA_SECRET", "dev-secret-please-change")
_SESSION_COOKIE = "pedi_session"
_SESSION_DURATION = 4 * 60 * 60  # 4 heures

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
# Database (historique patients)
# ---------------------------------------------------------------------------

def _get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Auth / sessions (simple)
# ---------------------------------------------------------------------------

def _hash_password(password: str) -> str:
    """Hash du mot de passe (PBKDF2 + sel aléatoire)."""
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return base64.b64encode(salt + dk).decode()


def _verify_password(password: str, hashed: str) -> bool:
    try:
        raw = base64.b64decode(hashed.encode())
        salt, dk = raw[:16], raw[16:]
        expected = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
        return hmac.compare_digest(expected, dk)
    except Exception:
        return False


def _make_session_token(username: str) -> str:
    """Génère un token signé pour la session."""
    expires = int(time.time()) + _SESSION_DURATION
    payload = f"{username}|{expires}"
    signature = hmac.new(_SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(f"{payload}|{signature}".encode()).decode()


def _verify_session_token(token: str) -> str | None:
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


def _get_user(username: str) -> dict | None:
    """Retourne l'utilisateur si existant."""
    _init_db()
    with _get_db_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None


def _create_default_admin() -> None:
    """Crée l'utilisateur admin par défaut si absent."""
    _init_db()
    with _get_db_connection() as conn:
        exists = conn.execute("SELECT 1 FROM users WHERE username = ?", ("admin",)).fetchone()
        if not exists:
            conn.execute(
                "INSERT INTO users (username, password_hash, full_name) VALUES (?, ?, ?)",
                ("admin", _hash_password("admin123"), "Administrateur"),
            )


def _init_db() -> None:
    """Crée la base de données si elle n'existe pas."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                first_name TEXT,
                last_name TEXT,
                dob TEXT,
                sex TEXT,
                weight_kg REAL,
                height_cm REAL,
                heart_rate REAL,
                bp_systolic INTEGER,
                bp_diastolic INTEGER,
                respiratory_rate REAL,
                spo2 REAL,
                symptom_duration_hours REAL,
                pain_score REAL,
                appetite_loss TEXT,
                vomiting TEXT,
                diarrhea TEXT,
                hematuria TEXT,
                age REAL,
                body_temperature REAL,
                wbc_count REAL,
                crp REAL,
                neutrophil_percentage REAL,
                appendix_diameter REAL,
                lower_right_abd_pain TEXT,
                migratory_pain TEXT,
                nausea TEXT,
                ipsilateral_rebound_tenderness TEXT,
                prob REAL,
                decision TEXT,
                risk_class TEXT,
                shap_b64 TEXT,
                raw_json TEXT
            )
            """,
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT
            )
            """,
        )
        # Create default admin user if missing
        conn.execute(
            "INSERT OR IGNORE INTO users (username, password_hash, full_name) VALUES (?, ?, ?)",
            ("admin", _hash_password("admin123"), "Administrateur"),
        )


def _save_record(record: dict) -> None:
    """Sauve une analyse dans la base de données."""
    _init_db()
    record = dict(record)
    record.setdefault("created_at", datetime.utcnow().isoformat())

    fields = [
        "created_at",
        "first_name",
        "last_name",
        "dob",
        "sex",
        "weight_kg",
        "height_cm",
        "heart_rate",
        "bp_systolic",
        "bp_diastolic",
        "respiratory_rate",
        "spo2",
        "symptom_duration_hours",
        "pain_score",
        "appetite_loss",
        "vomiting",
        "diarrhea",
        "hematuria",
        "age",
        "body_temperature",
        "wbc_count",
        "crp",
        "neutrophil_percentage",
        "appendix_diameter",
        "lower_right_abd_pain",
        "migratory_pain",
        "nausea",
        "ipsilateral_rebound_tenderness",
        "prob",
        "decision",
        "risk_class",
        "shap_b64",
        "raw_json",
    ]

    values = [record.get(f) for f in fields]
    placeholders = ",".join("?" for _ in fields)

    with _get_db_connection() as conn:
        conn.execute(
            f"INSERT INTO records ({','.join(fields)}) VALUES ({placeholders})",
            values,
        )


def _get_history(limit: int = 20) -> list[dict]:
    """Retourne les dernières analyses enregistrées."""
    _init_db()
    with _get_db_connection() as conn:
        cursor = conn.execute(
            "SELECT * FROM records ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        rows = cursor.fetchall()
    return [dict(r) for r in rows]


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
    """Affiche la page d'accueil."""
    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
        },
    )


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> HTMLResponse:
    """Page de connexion."""
    if _get_current_user(request):
        return RedirectResponse("/form", status_code=303)

    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
        },
    )


@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Traite la connexion et place un cookie de session."""
    user = _get_user(username)
    if not user or not _verify_password(password, user.get("password_hash", "")):
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "error": "Identifiant ou mot de passe incorrect.",
            },
            status_code=401,
        )

    token = _make_session_token(username)
    response = RedirectResponse(url="/form", status_code=303)
    response.set_cookie(_SESSION_COOKIE, token, httponly=True, samesite="lax")
    return response


@app.get("/logout")
async def logout() -> RedirectResponse:
    """Déconnecte l'utilisateur."""
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(_SESSION_COOKIE)
    return response


@app.get("/form", response_class=HTMLResponse)
async def read_form(request: Request) -> HTMLResponse:
    """Affiche le formulaire de saisie clinique avec section résultats."""
    user = _get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    return templates.TemplateResponse(
        "combined.html",
        {
            "request": request,
            "defaults": _defaults,
            "user": user,
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
    if not _get_current_user(request):
        return RedirectResponse("/login", status_code=303)

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
        "combined.html",
        {
            "request": request,
            "prob": f"{prob * 100:.1f}",
            "decision": decision,
            "risk_class": risk_class,
            "shap_b64": shap_b64,
            "form": form_data,
            "defaults": _defaults,
        },
    )


@app.post("/api/predict")
async def api_predict(request: Request) -> dict:
    """API pour prédiction — retourne JSON pour AJAX."""
    if not _get_current_user(request):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Non autorisé")

    form_values = await request.form()

    # Normalize values (FastAPI Form returns UploadFile or str)
    form_data: dict[str, str | float] = {
        k: v for k, v in form_values.items()
    }

    # Construire le DataFrame d'entrée (uniquement les champs attendus par le modèle)
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

    # Historique
    record = {
        **form_data,
        "prob": float(f"{prob * 100:.1f}"),
        "decision": decision,
        "risk_class": risk_class,
        "shap_b64": shap_b64,
        "raw_json": json.dumps(form_data, ensure_ascii=False),
    }
    try:
        _save_record(record)
    except Exception as exc:
        print(f"DB save error (non-fatal): {exc}")

    return {
        "prob": f"{prob * 100:.1f}",
        "decision": decision,
        "risk_class": risk_class,
        "shap_b64": shap_b64,
        "form": form_data,
    }


@app.get("/api/history")
async def api_history(request: Request, limit: int = 20, id: int | None = None) -> dict:
    """Retourne l'historique des analyses (dernières entrées).

    Si `id` est fourni, retourne uniquement l'enregistrement demandé.
    """
    if not _get_current_user(request):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Non autorisé")

    if id is not None:
        _init_db()
        with _get_db_connection() as conn:
            row = conn.execute("SELECT * FROM records WHERE id = ?", (id,)).fetchone()
        return {"records": [dict(row)] if row else []}

    return {"records": _get_history(limit)}


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
