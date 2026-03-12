"""
app/app.py
Application FastAPI — aide au diagnostic pédiatrique de l'appendicite.
Fournit une interface web pour entrer les paramètres cliniques d'un enfant
et obtenir une probabilité d'appendicite avec explication SHAP.
"""

from __future__ import annotations

import sys
import pathlib
import io
import base64

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# --- sys.path robustesse (lancé depuis n'importe quel répertoire)
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.train_model import load_model
from src.evaluate_model import compute_shap_values, predict_proba_safe

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PediAppendix",
    description="Aide au diagnostic pédiatrique de l'appendicite — Random Forest + SHAP",
    version="1.0.0",
)

_APP_DIR = pathlib.Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(_APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(_APP_DIR / "templates"))

# ---------------------------------------------------------------------------
# Chargement du modèle au démarrage (singleton)
# ---------------------------------------------------------------------------
_MODEL_DIR = _ROOT / "models"

_model = None
_feature_cols: list[str] = []
_defaults: dict[str, float] = {}


def _load_resources() -> None:
    """Charge le modèle RF et les valeurs par défaut des features."""
    global _model, _feature_cols, _defaults
    _model = load_model(_MODEL_DIR / "random_forest.joblib")
    test_data = joblib.load(_MODEL_DIR / "test_data.joblib")
    _feature_cols = test_data["feature_cols"]
    X_test: pd.DataFrame = test_data["X_test"]
    # Médiane du test set comme valeur par défaut (raisonnable pour la démo)
    _defaults = X_test.median().to_dict()


# Chargement synchrone au démarrage
_load_resources()


# ---------------------------------------------------------------------------
# Encodage des variables catégorielles (doit correspondre au LabelEncoder
# utilisé lors de l'entraînement — tri alphabétique).
# ---------------------------------------------------------------------------
_YES_NO = {"yes": 1, "no": 0}
_SEX_MAP = {"female": 0, "male": 1}
_STOOL_MAP = {"constipation": 0, "diarrhea": 1, "mucous_stool": 2, "normal": 3}
_RBC_URINE_MAP = {"no": 0, "few": 1, "moderate": 2, "many": 3}


def _encode_form(data: dict) -> pd.DataFrame:
    """
    Construit un vecteur de features à partir du formulaire HTML.

    1. Remplit toutes les features avec les valeurs par défaut du test set.
    2. Écrase avec les valeurs du formulaire (correctement encodées).
    3. Ajoute CRP_log depuis CRP si fourni.

    Returns
    -------
    DataFrame à une ligne, colonnes = _feature_cols.
    """
    row = dict(_defaults)  # copie des defaults

    # --- Numériques
    for field in ("Age", "BMI", "Weight", "Height", "Body_Temperature",
                  "WBC_Count", "CRP", "Hemoglobin", "Neutrophil_Percentage",
                  "Appendix_Diameter", "Thrombocyte_Count", "RBC_Count",
                  "RDW", "Segmented_Neutrophils"):
        val = data.get(field.lower(), "")
        if val != "" and val is not None:
            try:
                row[field] = float(val)
            except ValueError:
                pass  # garde le défaut

    # --- CRP_log (feature engineerée)
    if "CRP" in row:
        row["CRP_log"] = float(np.log1p(row["CRP"]))

    # --- Catégorielles yes/no
    for field in ("Migratory_Pain", "Lower_Right_Abd_Pain",
                  "Ipsilateral_Rebound_Tenderness",
                  "Contralateral_Rebound_Tenderness",
                  "Coughing_Pain", "Nausea", "Loss_of_Appetite", "Dysuria",
                  "Neutrophilia", "Peritonitis", "Psoas_Sign",
                  "Appendix_on_US", "US_Performed", "Free_Fluids",
                  "RBC_in_Urine", "Ketones_in_Urine", "WBC_in_Urine"):
        val = data.get(field.lower(), "")
        if val in _YES_NO:
            row[field] = _YES_NO[val]

    # --- Sexe
    sex = data.get("sex", "")
    if sex in _SEX_MAP:
        row["Sex"] = _SEX_MAP[sex]

    # --- Selles
    stool = data.get("stool", "")
    if stool in _STOOL_MAP:
        row["Stool"] = _STOOL_MAP[stool]

    # --- Sparse binaires (0/1) — non modifiées par l'utilisateur → déjà à 0
    # (la plupart de ces colonnes valent 0 par défaut dans le dataset)

    # Construire le DataFrame avec uniquement les features du modèle
    X = pd.DataFrame([{col: row.get(col, _defaults.get(col, 0)) for col in _feature_cols}])
    return X


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    age: float = Form(default=10.0),
    sex: str = Form(default="male"),
    bmi: str = Form(default=""),
    body_temperature: float = Form(default=37.5),
    wbc_count: float = Form(default=11.0),
    crp: float = Form(default=10.0),
    hemoglobin: float = Form(default=12.5),
    neutrophil_percentage: float = Form(default=65.0),
    appendix_diameter: str = Form(default=""),
    us_performed: str = Form(default="yes"),
    appendix_on_us: str = Form(default="no"),
    migratory_pain: str = Form(default="no"),
    lower_right_abd_pain: str = Form(default="yes"),
    ipsilateral_rebound_tenderness: str = Form(default="no"),
    nausea: str = Form(default="no"),
    loss_of_appetite: str = Form(default="no"),
) -> HTMLResponse:
    """
    Route de prédiction.
    Accepte les données du formulaire clinique et retourne :
      - La probabilité d'appendicite
      - La décision (appendicite / pas d'appendicite)
      - Un graphique SHAP waterfall expliquant la prédiction
    """
    form_data = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "body_temperature": body_temperature,
        "wbc_count": wbc_count,
        "crp": crp,
        "hemoglobin": hemoglobin,
        "neutrophil_percentage": neutrophil_percentage,
        "appendix_diameter": appendix_diameter,
        "us_performed": us_performed,
        "appendix_on_us": appendix_on_us,
        "migratory_pain": migratory_pain,
        "lower_right_abd_pain": lower_right_abd_pain,
        "ipsilateral_rebound_tenderness": ipsilateral_rebound_tenderness,
        "nausea": nausea,
        "loss_of_appetite": loss_of_appetite,
    }

    X = _encode_form(form_data)
    prob = float(predict_proba_safe(_model, X)[0])
    decision = "appendicite" if prob >= 0.5 else "pas d'appendicite"
    risk_class = "danger" if prob >= 0.5 else "success"

    # Génération SHAP waterfall en mémoire (base64 pour l'HTML)
    shap_b64: str | None = None
    try:
        _, sv = compute_shap_values(_model, X)
        # SHAP pour un seul patient — extract values correctly
        if isinstance(sv, list) and len(sv) == 2:
            sv_vals = sv[1][0]
            ev = _model.estimators_[0].tree_.threshold  # fallback
        elif isinstance(sv, np.ndarray) and sv.ndim == 2:
            sv_vals = sv[0]
        elif isinstance(sv, np.ndarray) and sv.ndim == 1:
            sv_vals = sv
        else:
            sv_vals = np.array(sv).ravel()

        import shap
        from src.evaluate_model import compute_shap_values as _csv

        explainer, sv_full = _csv(_model, X)

        # expected_value — extraire la classe positive
        ev_raw = explainer.expected_value
        if isinstance(ev_raw, (list, np.ndarray)) and len(ev_raw) == 2:
            base_val = float(ev_raw[1])
        else:
            base_val = float(ev_raw)

        # sv_full peut être (1, n_features) ou (2, 1, n_features) etc.
        if isinstance(sv_full, np.ndarray):
            if sv_full.ndim == 2:
                sv_single = sv_full[0]  # (n_features,)
            elif sv_full.ndim == 1 and sv_full.dtype == object:
                sv_single = sv_full[1][0] if len(sv_full) == 2 else sv_full[0]
            else:
                sv_single = sv_full.ravel()[:len(_feature_cols)]
        elif isinstance(sv_full, list):
            sv_single = sv_full[1][0] if len(sv_full) == 2 else sv_full[0][0]
        else:
            sv_single = np.zeros(len(_feature_cols))

        fig, _ = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=sv_single,
                base_values=base_val,
                data=X.iloc[0].values,
                feature_names=_feature_cols,
            ),
            show=False,
        )
        plt.title("Explication SHAP — Facteurs influençant la prédiction")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        shap_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close("all")
    except Exception as exc:
        shap_b64 = None
        print(f"SHAP waterfall error (non-fatal): {exc}")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "prob": f"{prob * 100:.1f}",
        "decision": decision,
        "risk_class": risk_class,
        "shap_b64": shap_b64,
        "form": form_data,
    })
