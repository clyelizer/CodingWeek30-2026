# 03 — Évaluation individuelle et explications SHAP

## Contexte

Une fois le modèle de production sélectionné (Random Forest, AUC = 0.9287),
le module `evaluate_model.py` prend en charge **l'évaluation d'un patient
individuel** lors d'une prédiction en temps réel via l'interface web.

Il fournit trois éléments pour chaque prédiction :
1. **La probabilité d'appendicite** (0–100%)
2. **Les valeurs SHAP** expliquant pourquoi le modèle a prédit ce score
3. **Un graphique waterfall** visualisant la contribution de chaque variable

---

## Fonctions du module (`src/evaluate_model.py`)

### 1. `predict_proba_safe(pipeline, X) → float`

Prédit la probabilité d'appartenance à la classe 1 (appendicite) pour une
observation.

```python
proba = predict_proba_safe(pipeline, X_patient)
# → 0.73  (73% de probabilité d'appendicite)
```

**Garantie d'assertion :** `0.0 ≤ proba ≤ 1.0`  
La fonction lève une `AssertionError` si la probabilité sort de [0, 1] —
ce qui ne devrait jamais arriver avec un classifieur sklearn standard, mais
l'assertion protège contre toute corruption du modèle.

---

### 2. `compute_shap_values(pipeline, X) → (shap_values, base_value)`

Calcule les valeurs SHAP pour l'observation via `shap.TreeExplainer`.

**Principe :**  
SHAP (SHapley Additive exPlanations) décompose la prédiction du modèle en
contributions additives de chaque feature, garantissant :
- **Consistance** : si une feature contribue plus dans un modèle, sa valeur SHAP est plus élevée
- **Null player** : une feature sans influence a une valeur SHAP = 0
- **Additivité** : `base_value + Σ(shap_i) = proba_prédite`

**Implémentation :**
```
base_value (prédiction moyenne sur le train set)
  + SHAP(Age = 8 ans)              → -0.12  (diminue le risque)
  + SHAP(CRP = 45 mg/L)            → +0.18  (augmente le risque)
  + SHAP(WBC = 18 G/L)             → +0.15  (augmente le risque)
  + SHAP(Lower_Right_Abd_Pain = 1) → +0.09  (augmente le risque)
  + ...
  = proba finale
```

**Décision technique — import lazy de `shap` :**  
`shap` est importé **à l'intérieur des fonctions** et non au niveau du module.
Sur Python 3.14, la chaîne d'import `shap → scipy → ...` est très lente et
provoque des `KeyboardInterrupt` au démarrage de l'application.
L'import lazy retarde ce coût jusqu'au premier appel réel.

**Compatibilité multi-versions SHAP :**  
Les versions récentes de SHAP retournent soit une liste `[sv_cls0, sv_cls1]`,
soit un array 3D `(n_obs, n_features, n_classes)`. La fonction gère les deux cas.

**Garantie d'assertion :** `len(shap_values) == n_features` (10)

---

### 3. `make_shap_waterfall_b64(shap_values, base_value, X_row) → str | None`

Génère le graphique waterfall SHAP et le retourne encodé en **PNG base64**,
directement injectable dans le HTML : `<img src="data:image/png;base64,...">`

**Format du graphique :**
- Axe horizontal : valeur de probabilité
- Barres rouges : features augmentant le risque d'appendicite
- Barres bleues : features diminuant le risque
- Départ : `base_value` (probabilité moyenne du modèle sur l'ensemble d'entraînement)
- Arrivée : probabilité prédite pour ce patient

**Comportement en cas d'erreur :**  
La fonction est **non-fatale** — retourne `None` si la génération échoue.
L'interface web affiche alors un message d'avertissement sans planter.
Cela évite qu'une incompatibilité de version SHAP/matplotlib ne rende
l'application inaccessible.

---

### 4. `build_results_summary(results, best_name) → pd.DataFrame`

Construit un tableau récapitulatif des métriques de tous les modèles,
utilisé pour l'affichage dans l'interface et la production de rapports.

```
        model    roc_auc      f1  accuracy   best
0  random_forest   0.9287  0.8457    0.8526   True
1  gradient_boosting 0.9141 0.8178  0.8269  False
2  logistic_regression 0.8283 0.7354  0.7564 False
3         svm   0.8102  0.7198    0.7436  False
```

---

## Interprétation SHAP — exemple type

Pour un patient avec appendicite probable (proba ~ 75%) :

| Feature | Valeur patient | ΔContribution |
|---------|---------------|---------------|
| `CRP` | 78 mg/L (élevé) | +0.22 ↑ |
| `WBC_Count` | 17.3 G/L (élevé) | +0.16 ↑ |
| `Appendix_Diameter` | 9.2 mm (> 6mm seuil clinique) | +0.13 ↑ |
| `Lower_Right_Abd_Pain` | Oui | +0.10 ↑ |
| `Ipsilateral_Rebound_Tenderness` | Oui | +0.08 ↑ |
| `Neutrophil_Percentage` | 82% (élevé) | +0.06 ↑ |
| `Age` | 7 ans | -0.05 ↓ |
| `Body_Temperature` | 37.8°C (légère fièvre) | +0.03 ↑ |
| `Migratory_Pain` | Non | -0.02 ↓ |
| `Nausea` | Oui | +0.01 ↑ |

La valeur de base (~0.41, correspondant à la prévalence de l'appendicite dans
le jeu d'entraînement) monte à ~0.73 sous l'effet cumulé de ces contributions.

---

## Tests unitaires (`tests/test_evaluate_model.py`) — 8 tests, tous passent

| Test | Fonction testée | Assertion |
|------|----------------|-----------|
| `test_predict_proba_safe_returns_float` | `predict_proba_safe` | `isinstance(result, float)` |
| `test_predict_proba_safe_in_range` | `predict_proba_safe` | `0.0 ≤ proba ≤ 1.0` |
| `test_predict_proba_safe_single_row` | `predict_proba_safe` | fonctionne avec 1 ligne |
| `test_predict_proba_safe_different_rows_can_differ` | `predict_proba_safe` | patients différents → probas différentes |
| `test_build_results_summary_correct_row_count` | `build_results_summary` | `len(df) == len(results)` |
| `test_build_results_summary_sorted_by_auc` | `build_results_summary` | `df.roc_auc` décroissant |
| `test_build_results_summary_best_column` | `build_results_summary` | colonne `best` présente |
| `test_build_results_summary_best_is_marked` | `build_results_summary` | `df[df.best].model[0] == best_name` |

**Note :** les tests SHAP (`compute_shap_values`, `make_shap_waterfall_b64`)
ne sont pas inclus dans la suite principale car l'import de `shap` est trop
lent sur Python 3.14 en CI. Ils sont marqués `pytest.mark.shap` et peuvent
être exécutés séparément : `pytest -m shap`.

```
pytest tests/test_evaluate_model.py -v
8 passed in ~1.1s
```

---

## Bilan des 3 modules (`tests/`)

```
pytest tests/ --rootdir="..." -q

tests/test_data_processing.py  11 passed
tests/test_model.py            15 passed
tests/test_evaluate_model.py    8 passed
──────────────────────────────────────────
TOTAL                          34 passed
```
