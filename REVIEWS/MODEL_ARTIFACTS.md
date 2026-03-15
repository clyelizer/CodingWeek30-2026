# Modèles & Artefacts — Revue détaillée

Fichiers analysés: `models/Random_Forest.pkl`, `models/preprocessor.pkl`, `models/best_model_info.pkl`, `src/train_model.py`, `src/evaluate_model.py`.

Objectif: vérifier sécurité, format, gestion des versions et recommandations
pour production (mise à jour de modèle, reproductibilité, signatures).

---

1) Contenu des artefacts
- Observé : le dossier `models/` contient plusieurs artefacts pickle/joblib
  (`Random_Forest.pkl`, `preprocessor.pkl`, `best_model_info.pkl`).
  Ces artefacts sont consommés par `app/app.py` via joblib.

2) Versioning et métadonnées
- `train_model.train_and_save_best_model()` écrit `best_model_info.json`
  (dans `train_model.py`), mais dans le repo il y a `best_model_info.pkl`.
  Confusion entre .json/.pkl est possible ; normaliser format est
  important (préférer JSON pour metadata lisible).

3) Gestion des mises à jour de modèle
- Recommandation :
  - Inclure un fichier `models/META.json` contenant :
    - `model_name`, `version` (semver), `created_at`, `metrics` (AUC, F1), `feature_cols`, `preprocessor_version`.
  - Lorsque le modèle évolue, incrémenter version et archiver ancien artefact
    (ex. `random_forest_v1.0.0.joblib`).

4) Validation au load
- Constat : `_load_model_artifact()` dans `app/app.py` vérifie la présence
  des méthodes `predict` et `predict_proba` (bon), mais ne vérifie pas
  la compatibilité des `feature_cols` avec le préprocesseur.

Recommandation :
  - Après chargement, valider que `preprocessor` ou `feature_cols` attendus
    correspondent au `X` fourni par l'API (nombre de colonnes, noms).
  - Charger et valider `best_model_info` metadata (version), refuser le
    démarrage si mismatch critique.

5) Sécurité (déjà repris dans SECURITY.md)
- Ne pas charger des artefacts non signés en production. Si signature
  impossible, au moins vérifier checksum/empreinte avant chargement.

6) Tests pour artefacts
- Ajouter tests unitaires qui :
  - vérifient que `models/preprocessor.pkl` peut transformer un DataFrame
    d'exemple et retourne le bon nombre de features.
  - test de compatibilité `model.predict_proba` avec vecteur d'entrée
    construit à partir de `feature_cols`.

---

Priorité immédiate (3 actions):
1. Normaliser métadonnées modèle en JSON (`models/META.json`).
2. Ajouter validation runtime du `feature_cols` au démarrage de l'app.
3. Ajouter test unitaire de smoke-load pour artefacts (dans tests/).
