# 03 — Évaluation individuelle et explications SHAP

## Contexte

Le module `evaluate_model.py` permet d'évaluer un patient individuel en temps réel via l'interface web.

Il fournit :
1. **La probabilité d'appendicite**
2. **Les valeurs SHAP** pour l'explicabilité
3. **Un graphique waterfall** PNG base64

---

## Fonctions du module (`src/evaluate_model.py`)

### 1. `predict_proba_safe`
Prédit la probabilité pour une observation.

### 2. `compute_shap_values`
Calcule les contributions SHAP. Utilise un **import lazy** de `shap` pour optimiser les performances.

### 3. `make_shap_waterfall_b64`
Génère le graphique waterfall encodé en base64 pour intégration directe dans le HTML.

---

## Interprétation SHAP

SHAP décompose la prédiction en contributions positives (rouge) ou négatives (bleu) par rapport à une valeur de base.

---

## Tests unitaires (`tests/test_evaluate_model.py`)

*Le module de tests pour l'évaluation individuelle est actuellement en attente d'implémentation ou vide.*
py            15 passed
tests/test_evaluate_model.py    8 passed
──────────────────────────────────────────
TOTAL                          34 passed
