# PediAppendix — Aide au diagnostic pédiatrique de l'appendicite

## Présentation

**PediAppendix** est un système d'aide à la décision clinique pour le diagnostic
de l'appendicite pédiatrique.

À partir de **10 paramètres cliniques courants** (examen physique, biologie,
échographie), il prédit la probabilité d'appendicite et fournit une explication
SHAP détaillée de chaque prédiction.

**Dataset :** Regensburg Pediatric Appendicitis (UCI), n = 776 patients.  
**Modèle :** Random Forest — AUC-ROC = **0.9287** sur le jeu de test (n = 156).

---

## Architecture du projet

```
projet/
├── data/
│   ├── raw/          data_finale.xlsx         (776 patients, 27 variables)
│   └── processed/    processed_data.joblib    (split train/test stratifié 80/20)
├── models/
│   ├── random_forest.joblib        ← modèle de production (AUC 0.9287)
│   ├── gradient_boosting.joblib    (AUC 0.9141)
│   ├── logistic_regression.joblib  (AUC 0.8283)
│   └── svm.joblib                  (AUC 0.8102)
├── src/
│   ├── data_processing.py   pipeline de traitement des données
│   ├── train_model.py       entraînement et évaluation des modèles
│   └── evaluate_model.py    prédiction individuelle + explications SHAP
├── tests/
│   ├── test_data_processing.py   11 tests
│   ├── test_model.py             15 tests
│   └── test_evaluate_model.py    8 tests        → 34 tests total, tous passent
├── app/
│   ├── app.py               interface FastAPI
│   └── templates/
│       ├── landing_page.html      page d'accueil
│       ├── auth.html              page de connexion
│       └── diagnosis_console.html console diagnostic (saisie + résultat + historique)
├── notebooks/
│   └── eda.ipynb            analyse exploratoire du dataset
├── MD/
│   ├── README.md            index de la documentation
│   ├── 01_data_processing.md
│   ├── 02_train_model.md
│   ├── 03_evaluate_model.md
│   └── 04_webapp.md
├── conftest.py              configuration pytest (sys.path)
├── requirements.txt
└── Dockerfile
```

---

## Features du modèle (10)

| # | Variable | Type | Source clinique |
|---|----------|------|----------------|
| 1 | `Lower_Right_Abd_Pain` | Binaire (oui/non) | Examen clinique |
| 2 | `Migratory_Pain` | Binaire (oui/non) | Examen clinique |
| 3 | `Ipsilateral_Rebound_Tenderness` | Binaire (oui/non) | Examen clinique |
| 4 | `Nausea` | Binaire (oui/non) | Examen clinique |
| 5 | `Body_Temperature` | Numérique (°C) | Examen clinique |
| 6 | `WBC_Count` | Numérique (G/L) | Biologie |
| 7 | `Neutrophil_Percentage` | Numérique (%) | Biologie |
| 8 | `CRP` | Numérique (mg/L) | Biologie |
| 9 | `Appendix_Diameter` | Numérique (mm) | Échographie |
| 10 | `Age` | Numérique (années) | Démographique |

---

## Résultats

| Modèle | AUC-ROC | F1 (macro) | Accuracy |
|--------|---------|------------|----------|
| **Random Forest** ← retenu | **0.9287** | **0.8457** | **0.8526** |
| Gradient Boosting | 0.9141 | 0.8178 | 0.8269 |
| Logistic Regression | 0.8283 | 0.7354 | 0.7564 |
| SVM (RBF) | 0.8102 | 0.7198 | 0.7436 |

**Interprétation de l'AUC = 0.9287 :** en tirant aléatoirement un patient positif
et un patient négatif, le modèle attribue une probabilité plus élevée au positif
dans 92.87% des cas.

---

## Installation

Python recommande : **3.11.x** (version cible du projet pour minimiser les incompatibilites binaires).

```bash
pip install -r requirements.txt
```

---

## Utilisation

### 1. Traitement des données (une seule fois)
```bash
python src/data_processing.py
# → data/processed/processed_data.joblib
```

### 2. Entraînement des modèles (une seule fois)
```bash
python src/train_model.py
# → models/random_forest.joblib (+ 3 autres modèles)
```

### 3. Tests unitaires
```bash
python -m pytest tests/ --rootdir="." -q
# 34 passed
```

### 4. Lancement de l'application web
```bash
uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
# → http://localhost:8000
```

### Docker
```bash
docker build -t pediappendix .
docker run -p 8000:8000 pediappendix
```

---

## Paradigme de développement

Ce projet suit un paradigme **fonctionnel strict** :
- **Une fonction = une tâche précise et testable**
- **Un test = une fonction = une assertion**
- Pas d'état global mutable entre fonctions
- Pas de data leakage : StandardScaler encapsulé dans chaque Pipeline sklearn

---

## Documentation technique

Voir le dossier [`MD/`](MD/README.md) pour la documentation détaillée
de chaque module avec les sorties et décisions de conception.

---

> ⚠️ **Avertissement médical** — Cet outil est à usage expérimental uniquement.
> Il ne remplace pas le jugement clinique d'un professionnel de santé.
> Dataset : Regensburg Pediatric Appendicitis — UCI Machine Learning Repository.
