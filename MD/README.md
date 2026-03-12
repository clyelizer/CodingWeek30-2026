# Documentation technique — PediAppendix

## Index des modules

| # | Fichier | Module | Contenu |
|---|---------|--------|---------|
| 01 | [01_data_processing.md](01_data_processing.md) | `src/data_processing.py` | Pipeline de traitement des données brutes |
| 02 | [02_train_model.md](02_train_model.md) | `src/train_model.py` | Entraînement des 4 modèles ML + résultats |
| 03 | [03_evaluate_model.md](03_evaluate_model.md) | `src/evaluate_model.py` | Prédiction individuelle + explications SHAP |
| 04 | [04_webapp.md](04_webapp.md) | `app/app.py` | Interface web FastAPI |

---

## Vue d'ensemble du pipeline

```
data/raw/data_finale.xlsx
        │
        ▼  src/data_processing.py
data/processed/processed_data.joblib
  (X_train 620×10, X_test 156×10, y_train, y_test)
        │
        ▼  src/train_model.py
models/
  random_forest.joblib       AUC = 0.9287  ← MEILLEUR
  gradient_boosting.joblib   AUC = 0.9141
  logistic_regression.joblib AUC = 0.8283
  svm.joblib                 AUC = 0.8102
        │
        ▼  src/evaluate_model.py  +  app/app.py
http://localhost:8000
  → formulaire 10 features
  → probabilité d'appendicite
  → graphique SHAP waterfall
```

---

## Résultats clés

### Jeu de données
- **776 patients** pédiatriques (Regensburg Pediatric Appendicitis, UCI)
- **27 variables** brutes → **10 features** retenues
- **Cible :** `Diagnosis` — 0 = pas d'appendicite (461, 59.4%), 1 = appendicite (315, 40.6%)
- **Split :** 80/20 stratifié → train : 620 patients, test : 156 patients

### Performances sur le jeu de test (n = 156)

| Modèle | AUC-ROC | F1 macro | Accuracy |
|--------|---------|----------|----------|
| **Random Forest** | **0.9287** | **0.8457** | **0.8526** |
| Gradient Boosting | 0.9141 | 0.8178 | 0.8269 |
| Logistic Regression | 0.8283 | 0.7354 | 0.7564 |
| SVM (RBF) | 0.8102 | 0.7198 | 0.7436 |

### Tests unitaires

| Fichier | Tests | Statut |
|---------|-------|--------|
| `tests/test_data_processing.py` | 11 | ✅ passent |
| `tests/test_model.py` | 15 | ✅ passent |
| `tests/test_evaluate_model.py` | 8 | ✅ passent |
| **Total** | **34** | **✅ 34/34** |

---

## Décisions techniques transversales

**Paradigme fonctionnel strict**
> Une fonction = une tâche = un test = une assertion précise.  
> Chaque fonction reçoit ses entrées et retourne ses sorties explicitement,
> sans effet de bord ni état global.

**Pas de data leakage**
> Le `StandardScaler` est encapsulé dans chaque `sklearn.Pipeline` et fitted
> exclusivement sur `X_train`. Il ne voit jamais `X_test`.

**Import lazy de SHAP**
> Sur Python 3.14, la chaîne `shap → scipy → ...` est très lente à l'import (~15s)
> et peut lever `KeyboardInterrupt`. `shap` est donc importé à l'intérieur des
> fonctions, uniquement lors du premier appel.

**Gestion du déséquilibre de classes**
> 59.4% négatifs / 40.6% positifs — corrigé par `class_weight="balanced"` sur
> Logistic Regression, Random Forest et SVM. Pour Gradient Boosting, la pondération
> intrinsèque des erreurs atténue partiellement ce déséquilibre.
