# Projet — Revue rapide et constats

Ce document liste problèmes, risques et points d'amélioration identifiés
en parcourant le code (lecture statique) — je note, je ne corrige pas.

## Contexte rapide
- Répertoire exploré : racine du projet (app/, src/, tests/, data/, MD/)
- Objectif : repérer points fragiles, erreurs probables et risques de
  maintenance ou de sécurité.

---

## Observations principales

- **Import-time side-effects (app/app.py)** : `app._load_resources()` est
  appelé au niveau module lors de l'import (`_load_resources()` à la fin du
  fichier). Risques : plantage d'import si `models/` ou `data/processed/`
  manquent, ralentissement des tests, effets de bord lors du rechargement.
  Fichier : [app/app.py](app/app.py)

- **Mismatch possible des types de `processed_data`** : dans
  `app._load_resources()` le code fait `processed = joblib.load(...)
  _defaults = processed["X_test"].median().to_dict()` — ceci suppose que
  `processed['X_test']` est un DataFrame (avec `.median()`), alors que le
  pipeline de `src/data_processing.py` produit typiquement des arrays numpy
  après transformation. Risque d'AttributeError/plantage à l'import.
  Fichiers : [app/app.py](app/app.py), [src/data_processing.py](src/data_processing.py)

- **Chargement d'artefacts par joblib (sécurité)** : `joblib.load` est utilisé
  pour charger des modèles et des artefacts. Charger des artefacts non
  fiables peut exécuter du code arbitraire — documenter et vérifier les
  sources des fichiers (ou signer/valider) si usage en production.
  Fichiers : [app/app.py](app/app.py), [src/train_model.py](src/train_model.py)

- **Administration par défaut codée en dur** : création d'un `admin` avec
  mot de passe `admin123` dans `_init_db()` de `app/app.py`. Pratique pour
  dev/tests, mais risque de sécurité si déployé tel quel.
  Fichier : [app/app.py](app/app.py)

- **Log/erreurs non structurés** : plusieurs `print(...)` pour erreurs
  non-fatal (SHAP, DB save) dans `app.app`. Préférer `logging` pour
  niveaux, et éviter prints en production.
  Fichier : [app/app.py](app/app.py)

- **Import/chemins relatifs inconsistants** : `src/train_model.py` modifie
  `sys.path` et utilise des imports relatifs/fallbacks. Fonctionnel mais
  fragile selon le cwd d'exécution. Préférence : utiliser un contrôle
  explicite du chemin racine (comme fait dans `app.app` avec `_ROOT`).
  Fichier : [src/train_model.py](src/train_model.py)

- **get_feature_names fallback fragile** : `src/data_processing.get_feature_names`
  tente `preprocessor.get_feature_names_out()` puis fallback qui invoque
  `preprocessor.feature_names_in_` — sur certaines versions de scikit-learn
  ces attributs peuvent différer; vérifier robustesse multi-versions.
  Fichier : [src/data_processing.py](src/data_processing.py)

- **Accès aux fichiers via chemins relatifs** : plusieurs modules utilisent
  littéraux `'models'`, `'data/raw/...'`, etc. `app/app.py` calcule `_ROOT`
  proprement, mais d'autres scripts (ex. `src/train_model.py`,
  `src/data_processing.py`) s'appuient sur le cwd. Cela peut rendre le
  comportement dépendant du répertoire de lancement.

- **SHAP & async** : le calcul SHAP est exécuté inline dans les routes
  FastAPI (synchrones). SHAP peut être coûteux et bloquer le loop
  event (ou retarder la réponse). Considérer exécution en background
  ou timeout/queue pour non-bloquant.
  Fichier : [app/app.py](app/app.py), [src/evaluate_model.py](src/evaluate_model.py)

- **SQLite usage dans contexte web** : l'application ouvre des connexions
  SQLite à la volée (`sqlite3.connect`) — attention aux accès concurrents
  si déploiement multi-thread/process; vérifier stratégie (pool, WAL, etc.).
  Fichier : [app/app.py](app/app.py)

- **Tests** : la suite `tests/test_app.py` isole correctement l'environnement
  via `monkeypatch` et `tmp_path`, mais si l'import d'`app.app` plante à cause
  de ressources manquantes (models/data), les fixtures peuvent échouer
  avant patch. Les tests actuels contiennent des assertions utiles pour
  garantir le comportement des templates et des endpoints.
  Fichier : [tests/test_app.py](tests/test_app.py)

## Points secondaires / améliorations suggérées (non-exhaustif)

- Centraliser la configuration de chemins (ROOT) et l'utiliser partout
  (`src/*` et `app/*`) pour éviter les problèmes de cwd.
- Remplacer `print` par `logging` configuré au niveau application.
- Lors de la sérialisation des données traitées (`processed_data.joblib`),
  stocker explicitement des métadonnées (p.ex. `feature_cols`, `defaults`)
  pour éviter dépendance au type de `X_test` au moment du reload.
- Documenter la confiance requise pour les artefacts chargés (joblib/pickle)
  et prévoir checksums ou signatures pour production.
- Revoir l'usage de `n_jobs=-1` dans `train_model.build_models()` pour
  éviter saturation des runners CI / postes à ressources limitées.

---

## Fichiers lus lors de la revue (extraits importants)
- [app/app.py](app/app.py)
- [src/data_processing.py](src/data_processing.py)
- [src/train_model.py](src/train_model.py)
- [src/evaluate_model.py](src/evaluate_model.py)
- [src/utils.py](src/utils.py)
- [tests/test_app.py](tests/test_app.py)
- [tests/test_train_model.py](tests/test_train_model.py)
- [requirements.txt](requirements.txt)
- [MD/04_webapp.md](MD/04_webapp.md)

---

## Prochaines actions proposées (à confirmer)
- Transformer le chargement de ressources en lazy-load (éviter side-effects
  à l'import).
- Sauvegarder explicitement `defaults` lors du pipeline de traitement des
  données pour garantir compatibilité au reload.
- Remplacer `print` par `logging` et ajouter monitoring minimal pour erreurs.

Fin de la revue initiale.

---

## Ajouts — templates, helpers SHAP et tests

- **Templates riches et logique JS cliente** : `app/templates/diagnosis_console.html`
  contient beaucoup de logique JS côté client (preview temps réel, génération
  PDF via `html2pdf`, debounce, history) — attention aux surfaces suivantes :
  - L'UI envoie des `FormData` à `/api/predict` et s'attend à `shap_b64` en base64;
    s'il manque la clé ou si la taille du PNG est trop grande, l'affichage peut
    être lent ou provoquer OOM dans le navigateur.
  - La page utilise des ressources externes (CDN Bootstrap, fontawesome,
    html2pdf). Pour un déploiement air-gapped, prévoir copies locales ou
    fallback.
  - De nombreux champs sont requis HTML; cependant le serveur `POST /api/predict`
    accepte des formulaires partiels et applique des defaults côté serveur —
    valider les deux côtés pour cohérence.
  Fichier : [app/templates/diagnosis_console.html](app/templates/diagnosis_console.html)

- **Page de login simple mais sans protection anti-brute-force** : `auth.html`
  est un template minimal. L'API côté serveur crée un cookie HTTP-only signé,
  mais il n'y a pas de mécanisme de verrouillage ou de throttling côté login.
  Pour production, envisager rate limiting et protection contre bruteforce.
  Fichier : [app/templates/auth.html](app/templates/auth.html)

- **SHAP helpers robustes mais assumptions** : `src/shap_explanations.py` fait
  un bon travail de lazy-loading et sélection d'explainer, mais notez :
  - `_get_explainer` choisit `KernelExplainer` pour non-tree models et effectue
    `shap.sample(X_background, min(100, len(X_background)))` — si
    `X_background` est un numpy array (comme produit par preprocessor), `shap.sample`
    peut accepter ou non ce type selon la version; tester. Le fallback
    fonctionne mais il faut s'assurer de la compatibilité des types.
  - `compute_shap_values` utilise `explainer.shap_values(X_sample)` et
    `_extract_shap_values` pour sélectionner la classe; la logique couvre
    plusieurs formats (list, ndarray 3D). Tests existent pour ces cas.
  - Les fonctions `generate_shap_summary` et `plot_waterfall` impriment des
    messages et retournent dicts `status: skipped` lorsqu'incompatibles,
    ce qui est pratique pour intégration CI.
  Fichier : [src/shap_explanations.py](src/shap_explanations.py)

- **Tests de `src` valides et ciblés** :
  - `tests/test_data_processing.py` vérifie les comportements attendus
    (downcast types, sauvegarde du préprocesseur, formes des sorties).
    Il force le `sys.path` vers `src/` et couvre les principales fonctions.
  - `tests/test_shap_explanations.py` couvre les chemins sans shap et
    l'extraction de valeurs SHAP (liste, 3D array). Bon signal que
    la logique SHAP a été pensée pour compatibilité multi-formats.
  Fichiers : [tests/test_data_processing.py](tests/test_data_processing.py), [tests/test_shap_explanations.py](tests/test_shap_explanations.py)

## Recommandations rapides suite aux fichiers ajoutés

- Documenter les formats attendus pour `processed_data.joblib` (types: DataFrame vs ndarray)
  et sérialiser explicitement `feature_cols` et `defaults` au moment du pipeline.
- Ajouter des contrôles côté serveur sur la taille maximale d'image `shap_b64`
  acceptée pour éviter charges mémoire élevées côté client et côté serveur.
- Prévoir stratégie de rate-limiting sur `/login` et surveillance des tentatives.

