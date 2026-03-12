# PediAppendix — Pediatric Appendicitis Clinical Decision Aid

> A machine-learning web application that predicts the probability of appendicitis in pediatric patients and explains its reasoning with SHAP (SHapley Additive exPlanations).

[![CI](https://github.com/<YOUR_ORG>/CodingWeek/actions/workflows/ci.yml/badge.svg)](https://github.com/<YOUR_ORG>/CodingWeek/actions/workflows/ci.yml)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model Performance](#model-performance)
4. [Top SHAP Features](#top-shap-features)
5. [Project Structure](#project-structure)
6. [Quickstart](#quickstart)
7. [Running Tests](#running-tests)
8. [Docker Deployment](#docker-deployment)
9. [API Reference](#api-reference)

---

## Project Overview

PediAppendix is a clinical decision support tool built around the [UCI Regensburg Pediatric Appendicitis dataset](https://archive.ics.uci.edu/dataset/938/regensburg+pediatric+appendicitis). It:

- Preprocesses raw multi-source clinical data (labs, ultrasound, symptoms)
- Trains four ML models with a strict functional pipeline (no data leakage)
- Evaluates models with ROC-AUC, precision, recall, and F1
- Exposes a FastAPI web interface where clinicians can enter patient data and receive an instant prediction + SHAP waterfall explanation

---

## Dataset

| Property | Value |
|---|---|
| Source | UCI ML Repository — Regensburg Pediatric Appendicitis |
| Rows / columns | 776 patients × 59 features |
| Missing values | 0 (after preprocessing) |
| Target | `Diagnosis` — appendicitis (1) vs. no appendicitis (0) |
| Class balance | 463 positive / 313 negative (ratio 1.46 : 1) — **no resampling needed** |
| Train / Test split | 620 / 156 (80/20, stratified) |

**Preprocessing steps** (functional pipeline, zero side-effects):

1. `remove_leakage_cols` — drops outcome-leaking columns (Severity, Management, Alvarado score, …)
2. `remove_biological_impossibles` — removes physiologically impossible values (T < 34 °C, Hgb > 20 g/dL, …)
3. `winsorize_iqr` — clips outliers at 1.5 × IQR
4. `impute_numeric / impute_categorical` — median / mode imputation
5. `add_log_transform` — log1p transform for CRP (right-skewed)
6. `encode_sparse_binary` — binarise sparse binary columns
7. `optimize_memory` — downcast dtypes to reduce RAM

---

## Model Performance

All metrics evaluated on the held-out test set (n = 156).

| Model | Accuracy | Precision | Recall | F1 | **AUC** |
|---|---|---|---|---|---|
| Random Forest | 0.9038 | 0.890 | 0.957 | 0.922 | **0.9735** |
| CatBoost | 0.9103 | 0.901 | 0.940 | 0.920 | 0.9734 |
| LightGBM | 0.9295 | 0.920 | 0.950 | 0.935 | 0.9616 |
| SVM | 0.8654 | 0.855 | 0.915 | 0.884 | 0.9548 |

**Best model: Random Forest (AUC = 0.9735)**  
The app uses the Random Forest model for predictions.

---

## Top SHAP Features

Based on SHAP summary plot (mean |SHAP value| across test set):

1. **WBC_Count** — white blood cell count (strongest driver)
2. **Neutrophil_Percentage** — proportion of neutrophils
3. **CRP_log** — log-transformed C-reactive protein
4. **Age** — younger children have higher appendicitis risk
5. **Body_Temperature** — fever indicator
6. **Appendix_Diameter** (ultrasound) — > 6 mm is diagnostic
7. **Appendix_on_US** — appendix visible on ultrasound
8. **Migratory_Pain** — classic periumbilical-to-RLQ migration
9. **Lower_Right_Abd_Pain** — McBurney's point tenderness
10. **Ipsilateral_Rebound_Tenderness** — peritoneal irritation sign

---

## Project Structure

```
CodingWeek/
├── app/
│   ├── app.py                 # FastAPI application (routes, form processing)
│   ├── main.py                # Uvicorn entry point
│   ├── static/plots/          # Generated ROC / SHAP plots
│   └── templates/
│       ├── index.html         # Clinical input form
│       └── result.html        # Prediction result + SHAP waterfall
├── data/
│   ├── external/              # Raw Excel files (not committed)
│   ├── processed/             # Cleaned dataset
│   └── raw/                   # Original downloads
├── models/                    # Serialised .joblib model files
├── notebooks/
│   └── eda.ipynb              # Exploratory data analysis
├── src/
│   ├── data_processing.py     # Pure preprocessing functions
│   ├── train_model.py         # Training pipeline
│   └── evaluate_model.py      # Metrics + SHAP visualisation
├── tests/
│   ├── test_data_processing.py  # 42 unit tests
│   └── test_model.py            # 25 unit tests
├── .github/workflows/ci.yml   # GitHub Actions CI pipeline
├── Dockerfile                 # Multi-stage production image
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/<YOUR_ORG>/CodingWeek.git
cd CodingWeek
pip install -r requirements.txt
```

### 2. Place raw data

Drop the Regensburg Pediatric Appendicitis Excel files into `data/external/`.

### 3. Run the full pipeline

```bash
# Preprocess + train all models (~2 min)
python -m src.train_model

# Generate evaluation plots
python -m src.evaluate_model
```

### 4. Start the web app

```bash
uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
```

Then open [http://localhost:8000](http://localhost:8000).

---

## Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

Expected: **67 tests passed** (42 data processing + 25 model tests).

---

## Docker Deployment

```bash
# Build
docker build -t pediappendix:latest .

# Run
docker run -p 8000:8000 pediappendix:latest
```

The image uses a multi-stage build (builder + slim runtime) to keep the final image lean.

---

## API Reference

### `GET /`

Returns the clinical input form (HTML).

### `POST /predict`

| Field | Type | Description |
|---|---|---|
| `age` | float | Patient age (years) |
| `sex` | `male` / `female` | Biological sex |
| `body_temperature` | float | Temperature (°C) |
| `wbc_count` | float | WBC (× 10³/µL) |
| `crp` | float | CRP (mg/L) |
| `hemoglobin` | float | Hgb (g/dL) |
| `neutrophil_percentage` | float | Neutrophil % |
| `appendix_diameter` | float | Ultrasound diameter (mm) |
| `us_performed` | `yes` / `no` | Ultrasound performed |
| `appendix_on_us` | `yes` / `no` | Appendix visible on US |
| `migratory_pain` | `yes` / `no` | Migratory pain |
| `lower_right_abd_pain` | `yes` / `no` | RLQ tenderness |
| `ipsilateral_rebound_tenderness` | `yes` / `no` | Rebound tenderness |
| `nausea` | `yes` / `no` | Nausea / vomiting |
| `loss_of_appetite` | `yes` / `no` | Anorexia |

**Returns**: HTML page with probability percentage, risk alert, and SHAP waterfall chart.

