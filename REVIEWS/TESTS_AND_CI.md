# Tests & Qualité — Revue détaillée

Fichiers analysés: `tests/`, `conftest.py`, `.github/workflows/ci.yml`.

Objectif: évaluer couverture test, robustesse des fixtures, rapidité et
propositions d'amélioration pour CI et tests unitaires.

---

1) Couverture et organisation
- Observé : plusieurs tests couvrent `src/` et `app/` (test_app.py, test_data_processing.py,
  test_shap_explanations.py, test_train_model.py, test_utils.py). Le CI exécute
  `pytest` et mesure couverture (`--cov=src`).

2) Fixtures & isolation
- `tests/test_app.py` utilise `monkeypatch` pour remplacer `_model`, `_defaults`
  et `_feature_cols` — bonne pratique pour isoler tests web du heavy model load.
- Cependant `app/app.py` exécute `_load_resources()` à l'import, ce qui peut
  provoquer side-effects avant que `monkeypatch` s'applique si l'import se
  produit plus tôt. Actuellement les tests ont contourné ce problème via
  reloading (importlib.reload) et fixtures, mais la meilleure solution est
  d'éviter le chargement au niveau module et passer à lazy-load.

3) Tests manquants / suggestions
- Ajouter tests pour :
  - compatibilité entre `models/preprocessor.pkl` et `feature_cols` (smoke-test),
  - endpoints d'API `/api/history` edge-cases (limit, id absent),
  - sécurité: s'assurer que login bloque tentatives invalides après N essais
    (si mechanism implemented).

4) Performance des tests
- Les tests font `joblib.dump()` et chargent artefacts temporaires — bien.
- Pour accélérer CI :
  - marquer tests lents / integration tests (xdist, markers),
  - paralléliser tests (`pytest -n auto` via xdist) si compatibles.

5) Reproductibilité locale
- `conftest.py` insère racine projet dans sys.path — utile, mais
  documenter la stratégie d'exécution des tests pour les nouveaux devs.

---

Priorité immédiate (3 actions):
1. Refactor `app/app.py` pour éviter import-time side-effects et faciliter tests.
2. Ajouter tests smoke-load pour `models/` artefacts.
3. Nettoyer le workflow CI et garantir que `pytest tests/` s'exécute dans
   un environnement contrôlé (python matrix).
