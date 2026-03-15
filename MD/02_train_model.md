# 02 — Entraînement et sélection des modèles

## Contexte

L'objectif est de trouver le modèle ML le plus performant pour prédire
l'appendicite pédiatrique (cible binaire `Diagnosis` 0/1).  
Tous les modèles sont entraînés sur le même split 80/20 stratifié produit par
`data_processing.py` et sauvegardé dans `data/processed/processed_data.joblib`.

**Jeu de données :**
- Entraînement : **620 patients** (247 positifs, 373 négatifs)
- Test : **156 patients** (63 positifs, 93 négatifs)

---

## Architecture — Pipeline sklearn

Chaque modèle est encapsulé dans un `sklearn.Pipeline` à deux étapes :

```
Pipeline
  ├── scaler : StandardScaler
  └── clf    : (classifieur)
```

**Justification architecturale :**  
L'encapsulation du `StandardScaler` dans le pipeline **garantit l'absence de
data leakage** : le `fit` du scaler est strictement limité à `X_train`. Si le
scaling avait été appliqué en amont (dans `data_processing.py`), le scaler
aurait vu les statistiques de `X_test`, biaisant les résultats.

---

## Modèles candidats (`src/train_model.py`)

| Modèle | Paramètres clés | Rôle |
|--------|----------------|------|
| **Logistic Regression** | `max_iter=1000`, `class_weight="balanced"` | Baseline linéaire interprétable |
| **Random Forest** | `n_estimators=200`, `class_weight="balanced"`, `random_state=42` | Modèle principal, supporte SHAP |
| **Gradient Boosting** | `n_estimators=200`, `learning_rate=0.05`, `max_depth=4` | Concurrent GB pour données tabulaires |
| **SVM (RBF)** | `class_weight="balanced"`, `probability=True` | Référence non-linéaire sans SHAP natif |

**Gestion du déséquilibre de classes :**  
Avec 59.4% de négatifs / 40.6% de positifs, un classificateur naïf atteindrait
59.4% d'accuracy sans rien apprendre. Le paramètre `class_weight="balanced"`
pondère les erreurs sur la classe minoritaire (positifs) pour corriger ce biais.

---

## Fonctions du module (`src/train_model.py`)

| Fonction | Entrée | Sortie | Tâche |
|----------|--------|--------|-------|
| `build_logistic_regression()` | — | `Pipeline` | Construit le pipeline LR |
| `build_random_forest()` | — | `Pipeline` | Construit le pipeline RF |
| `build_gradient_boosting()` | — | `Pipeline` | Construit le pipeline GB |
| `build_svm()` | — | `Pipeline` | Construit le pipeline SVM |
| `train_model(pipeline, X_train, y_train)` | Pipeline vierge, données | `Pipeline` fitté | Entraîne un pipeline |
| `evaluate_model(pipeline, X_test, y_test)` | Pipeline fitté, données test | `dict` métriques | Calcule AUC, F1, Accuracy |
| `select_best_model(results)` | dict métriques | `str` | Sélectionne le meilleur par AUC |
| `save_model(pipeline, name, dir)` | Pipeline fitté | `Path` | Sauvegarde `.joblib` |
| `load_model(path)` | `Path` | `Pipeline` | Charge et valide `.joblib` |
| `run_training(data_path, models_dir)` | chemins | `dict` | Pipeline complet end-to-end |

---

## Métriques choisies

**Métrique principale : AUC-ROC**  
L'AUC-ROC mesure la capacité de discrimination du modèle **indépendamment du
seuil de décision**. Dans un contexte médical où le seuil optimal peut varier
selon le praticien (trade-off sensibilité/spécificité), l'AUC est la métrique
de référence.

**Métrique secondaire : F1-score macro**  
Le F1 macro tient compte des deux classes équitablement, ce qui est important
face au déséquilibre 60/40.

---

## Résultats sur le jeu de test (n = 156)

| # | Modèle | AUC-ROC | F1 (macro) | Accuracy |
|---|--------|---------|------------|----------|
| 🥇 | **Random Forest** | **0.9287** | **0.8457** | **0.8526** |
| 🥈 | Gradient Boosting | 0.9141 | 0.8178 | 0.8269 |
| 🥉 | Logistic Regression | 0.8283 | 0.7354 | 0.7564 |
| 4 | SVM (RBF) | 0.8102 | 0.7198 | 0.7436 |

### Analyse des résultats

**Random Forest — meilleur modèle (AUC = 0.9287)**  
L'AUC de 0.9287 signifie qu'en tirant aléatoirement un patient positif et un
patient négatif, le modèle attribue une probabilité plus haute au positif dans
**92.87% des cas**. C'est une excellente capacité de discrimination pour un
contexte médical.

**Gradient Boosting (AUC = 0.9141)**  
Très proche du RF mais légèrement inférieur ici. Le GB est souvent favori sur
les données tabulaires médicales, mais le RF bénéficie de la pondération
`class_weight` que le GB sklearn standard ne supporte pas nativement.

**Logistic Regression (AUC = 0.8283)**  
La LR atteint une AUC correcte mais révèle les limites d'un modèle linéaire :
les interactions entre les features cliniques (ex. CRP élevé × leucocytose)
ne sont pas capturables sans feature engineering.

**SVM RBF (AUC = 0.8102)**  
Performance décevante comparée aux méthodes ensemblistes. Le SVM RBF est
sensible aux hyperparamètres (`C`, `gamma`) que nous n'avons pas tuné ici.

---

## Modèle retenu et fichiers sauvegardés

Le **Random Forest** est sélectionné comme modèle de production.

Tous les modèles sont sauvegardés pour comparaison :

```
models/
  random_forest.joblib       ← modèle de production
  gradient_boosting.joblib
  logistic_regression.joblib
  svm.joblib
```

---

## Tests unitaires (`tests/test_model.py`) — 15 tests, tous passent

| Test | Assertion |
|------|-----------|
| `test_build_logistic_regression_has_two_steps` | `len(steps) == 2` |
| `test_build_random_forest_has_two_steps` | `len(steps) == 2` |
| `test_build_gradient_boosting_has_two_steps` | `len(steps) == 2` |
| `test_build_svm_has_two_steps` | `len(steps) == 2` |
| `test_build_random_forest_last_step_is_clf` | `steps[-1][0] == "clf"` |
| `test_train_model_returns_pipeline` | `isinstance(fitted, Pipeline)` |
| `test_train_model_fitted_can_predict` | `len(preds) == len(y)` |
| `test_evaluate_model_roc_auc_in_range` | `0.0 <= auc <= 1.0` |
| `test_evaluate_model_returns_three_keys` | `keys == {roc_auc, f1, accuracy}` |
| `test_evaluate_model_accuracy_in_range` | `0.0 <= acc <= 1.0` |
| `test_select_best_model_returns_highest_auc` | `best == "rf"` sur données connues |
| `test_select_best_model_name_in_results` | `best in results` |
| `test_save_model_creates_file` | `out.exists()` après sauvegarde |
| `test_load_model_returns_pipeline` | `isinstance(loaded, Pipeline)` |
| `test_load_model_can_predict` | `len(preds) == len(y)` après rechargement |

```
pytest tests/test_model.py -v
15 passed in ~6.2s
```
