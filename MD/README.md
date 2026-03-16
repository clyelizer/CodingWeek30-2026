# Documentation technique — PediAppendix

## Index des modules

| # | Fichier | Module | Contenu |
|---|---------|--------|---------|
| 01 | [01_data_processing.md](01_data_processing.md) | `src/data_processing.py` | Pipeline de traitement des données brutes |
| 02 | [02_train_model.md](02_train_model.md) | `src/train_model.py` | Entraînement des modèles ML + résultats |
| 03 | [03_evaluate_model.md](03_evaluate_model.md) | `src/evaluate_model.py` | Prédiction individuelle + explications SHAP |
| 04 | [04_webapp.md](04_webapp.md) | `app/app.py` | Interface web FastAPI |

---

## Vue d'overview du pipeline

```
data/raw/dataset.xlsx
        │
        ▼  src/data_processing.py
data/processed/processed_data.joblib
  (X_train 620×10, X_test 156×10, y_train, y_test)
        │
        ▼  src/train_model.py
models/
  preprocessor.pkl           ← Préprocesseur fitté
  random_forest_model.pkl    ← Modèle Random Forest
  gradient_boosting_model.pkl
  ...
        │
        ▼  src/evaluate_model.py  +  app/app.py
http://localhost:8000
  → Authentification / Landing Page
  → Formulaire 10 features
  → Probabilité d'appendicite
  → Graphique SHAP waterfall
```

---

## Résultats clés

### Jeu de données
- **776 patients** pédiatriques (Regensburg Pediatric Appendicitis, UCI)
- **27 variables** brutes → **10 features** retenues
- **Cible :** `Diagnosis` — 0 = pas d'appendicite, 1 = appendicite
- **Split :** 80/20 stratifié → train : 620 patients, test : 156 patients

### Performances du modèle Random Forest
- **AUC-ROC :** ~0.92
- **Précision globale :** ~0.85

### Tests unitaires (GitHub Actions)

| Fichier | Tests | Statut |
|---------|-------|--------|
| `tests/test_data_processing.py` | 11 | ✅ Passent |
| `tests/test_model.py` | 2 | ✅ Passent |
| **Total** | **13** | **✅ 13/13** |

---

## Décisions techniques transversales

**Paradigme fonctionnel**
> Une fonction = une tâche = un test = une assertion précise.  
> Chaque fonction reçoit ses entrées et retourne ses sorties explicitement,
> sans effet de bord ni état global.

**Pas de data leakage**
> Le `StandardScaler` (via le préprocesseur) est fitted exclusivement sur `X_train`.

**Import lazy de SHAP**
> L'import de `shap` est différé pour ne pas ralentir le démarrage de l'application.
