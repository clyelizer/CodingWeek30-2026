# Données & Pipeline — Revue détaillée

Fichiers analysés: `src/data_processing.py`, `src/train_model.py`, `data/processed/processed_data.joblib`, `MD/01_data_processing.md`.

Objectif: vérifier robustesse du pipeline, formats persistés, reproductibilité,
gestion des valeurs par défaut et risques de data leakage.

---

1) Types et formats persistés
- Constat : `save_processed_data()` dans `src/data_processing.py` sérialise
  un dict contenant `X_train`, `X_test`, `y_train`, `y_test`, et `feature_cols`.
  Cependant dans `app/app.py` le code s'attend à pouvoir appeler
  `processed['X_test'].median().to_dict()`, supposant `X_test` un DataFrame.
- Observation : dans `load_and_preprocess()` le retour `X_train_processed`
  et `X_test_processed` sont des numpy arrays (après transformation). Si
  `save_processed_data()` enregistre des arrays, `.median()` n'existe.
- Recommandation :
  - Sérialiser explicitement `defaults` (median per feature) au moment où
    on a encore DataFrame/nom des colonnes. Sauvegarder `defaults` dict
    pour consommation par l'app.
  - Mettre `feature_cols` dans la meta et éviter de dépendre de types
    processés (DataFrame vs ndarray) dans le code d'inférence.

2) Robustesse des fonctions de preprocessing
- `get_feature_names()` tente `preprocessor.get_feature_names_out()` puis
  fallback qui appelle `preprocessor.feature_names_in_`. Selon version
  scikit-learn, le comportement varie.
- Recommandation :
  - Standardiser le préprocesseur (documenter version scikit-learn supportée),
  - Lors du run pipeline, calculer et sauvegarder `feature_cols` explicites
    (liste de noms) dans le joblib pour éviter tout heuristique au reload.

3) Data leakage & stratification
- Observé : `split_train_test()` utilise `train_test_split(..., stratify=y)`
  ce qui est bien. S'assurer que les transformations (imputer/scaler) sont
  uniquement fit sur le train (code actuel respecte cela via fit_transform).

4) Memory optimization
- `optimize_memory()` downcast les colonnes integer/float et convertit
  objet -> category si cardinalité faible. C'est utile mais modifie types
  et peut impacter pipelines qui s'attendent à types natifs.
- Recommandation :
  - Exécuter `optimize_memory()` avant sélection des colonnes et documenter
    que les downstream steps acceptent `category`.

5) Reproductibilité
- Recommandation :
  - Enregistrer versions de packages (pip freeze -> requirements-lock.txt)
  - Versionner `processed_data.joblib` ou stocker hash pour traçabilité
  - Ajouter un notebook ou script de vérification qui recharge `processed_data`
    et exécute un sanity-check (shapes, feature order).

6) Fichiers présents dans repo
- `data/processed/processed_data.joblib` existe — inspecter et vérifier
  qu'il contient bien `feature_cols` et, idéalement, `defaults`. Si non,
  régénérer via `src/run_pipeline` et commit de l'artefact.

---

Priorité immédiate (3 actions):
1. Modifier `save_processed_data()` pour inclure `defaults` (median dict).
2. Ajouter un script `scripts/verify_processed_data.py` qui valide le joblib
   (content types, feature_cols order).
3. Documenter le format `processed_data.joblib` dans MD/01_data_processing.md.
