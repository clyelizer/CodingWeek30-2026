# Audit approfondi — preuves & remédiations

Ce document rassemble preuves extraites automatiquement, analyses détaillées
et remédiations concrètes pour les points critiques identifiés dans le
dépôt. Il complète les fiches plus spécifiques déjà produites dans `REVIEWS/`.

---

1) Preuves (extraits grep)

Les recherches ont trouvé les motifs suivants (emplacements indiqués) :

- `joblib.load` / `joblib.dump` :
  - [src/train_model.py](src/train_model.py#L45) — chargement depuis chemin candidat
  - [src/train_model.py](src/train_model.py#L60) — fallback loading
  - [src/train_model.py](src/train_model.py#L122) — sauvegarde du modèle
  - [src/data_processing.py](src/data_processing.py#L209, L256) — sauvegarde du préprocesseur et processed_data
  - [app/app.py](app/app.py#L131) — chargement d'artefacts modèles
  - [app/app.py](app/app.py#L159) — chargement `processed_data.joblib`
  - Plusieurs tests utilisent `joblib.dump` / `joblib.load` pour fixtures ([tests/test_app.py], [tests/test_model.py]).

- Secrets & admin par défaut :
  - [app/app.py](app/app.py#L75) — `_SECRET_KEY = os.environ.get("PEDIA_SECRET", "dev-secret-please-change")`
  - [app/app.py](app/app.py#L281, L347) — création utilisateur `admin` avec mot de passe `admin123`
  - [tests/test_app.py](tests/test_app.py#L73) — test se connecte avec `admin123` (fixture reliant au comportement de dev).

- `except Exception:` généralisés / swallow errors :
  - [src/data_processing.py](src/data_processing.py#L226) — except Exception (optimisation / fallback)
  - [app/app.py](app/app.py#L132, L211, L244, L600, L642, L656) — plusieurs except Exception couvrant parsing, session decoding, SHAP, DB save.

- Cryptographie basique & usages :
  - [app/app.py](app/app.py#L207) — `base64.b64decode` et slicing pour sel+digest.
  - [app/app.py](app/app.py#L210-L211) — `hmac.compare_digest` usage correct pour éviter timing attacks.

---

2) Analyse détaillée et risques (par domaine)

2.1. Chargement d'artefacts (joblib/pickle)

- Description : le projet utilise `joblib` (pickle-based) pour sérialiser
  modèles, préprocesseur et jeux transformés. Les artefacts sont chargés
  automatiquement par `app/app.py` au moment de l'import.

- Risques :
  - Exécution de code arbitraire via charge utile pickle (RCE) si artefact compromis.
  - Plantage de l'application au démarrage si artefacts manquants ou incompatibles.
  - Tests et CI peuvent dépendre d'artefacts présents dans le repo — ceci
    masque des problèmes de packaging / reproductibilité.

- Remédiations :
  1. Restreindre origine des artefacts : n'accepter que des artefacts signés
     (HMAC/GPG) ou provenant d'un stockage contrôlé.
  2. Ajouter validation après `joblib.load` : vérifier interface (predict/predict_proba), vérifier `feature_cols` et metadata.
  3. Éviter `joblib.load` sur import : déplacer le chargement en `@app.on_event('startup')`
     ou utiliser lazy-loading (getter qui charge au premier usage).
  4. Préférer formats plus sûrs pour distribution (ONNX, TorchScript, PMML) si possible.

2.2. Secrets et admin par défaut

- Description : clef secrète et compte admin par défaut sont présents
  (placeholder et `admin123`).

- Risques : compromission triviale des accès en environnement non configuré.

- Remédiations :
  1. Refuser démarrage en production si `PEDIA_SECRET` est le placeholder.
  2. Conditionner création d'admin par variable d'environnement dev-only.
  3. Documenter procédure d'initialisation des comptes et stockage des secrets.

2.3. Gestion d'exceptions trop large

- Description : plusieurs `except Exception:` attrapent toutes erreurs et
  retournent None ou font un `print`, ce qui cache potentiellement erreurs
  sérieuses (data corruption, incompatibilité d'artefact).

- Risques :
  - Masquage de bugs critiques qui doivent arrêter le process.
  - Difficulté de debug en production, manque de logs structurés.

- Remédiations :
  1. Remplacer `except Exception:` par exceptions ciblées (ValueError, FileNotFoundError, ImportError, etc.).
  2. Utiliser `logging.exception(...)` pour capturer stacktrace au lieu de `print`.
  3. Pour erreurs critiques (artifact incompatible), échouer au démarrage.

2.4. SHAP / performances

- Description : calcule SHAP inline dans les routes, renvoyant image base64.

- Risques :
  - Latence élevée / blocage de worker si calcul SHAP long.
  - Consommation mémoire (image base64) côté serveur et client.

- Remédiations :
  1. Exécuter SHAP en tâche background (FastAPI BackgroundTasks ou queue).
  2. Ajouter timeout/limit sur la génération d'image et renvoyer placeholder en cas d'échec.
  3. Compresser / réduire la taille de l'image avant encodage base64.

2.5. SQLite en contexte web

- Description : usage direct de sqlite via `sqlite3.connect` pour l'historique.

- Risques :
  - Concurrence (multi-workers/process) peut provoquer verrous et erreurs.

- Remédiations :
  1. Configurer SQLite en WAL mode et vérifier PRAGMA journal_mode.
  2. Pour déploiement scalable, migrer vers PostgreSQL ou autre DB réseau.

---

3) Liste de remédiations concrètes et priorisées (propositions de patch)

Priorité haute (sécurité & fiabilité) :

- A. Passer le chargement des ressources (models + processed_data) en lazy-load.
  - Implémentation : remplacer l'appel global `_load_resources()` par `@app.on_event('startup')` ou par fonction `_ensure_resources_loaded()` utilisée quand `_model` est None.
  - Tests à ajouter : test simulant absence de modèles et s'assurant que l'app démarre et que endpoints renvoient erreurs contrôlées.

- B. Refuser admin/default secret en production.
  - Implémentation : vérifier `if _SECRET_KEY == 'dev-secret-please-change' and not DEBUG: raise RuntimeError(...)` ou similaire. Ajouter variable `PEDIA_ENV=development|production`.

- C. Remplacer prints par logging et lever sur erreurs critiques.
  - Implémentation : config logging (module-level) et remplacer `print(f"SHAP error (non-fatal): {exc}")` par `logger.exception('SHAP error')`.

Priorité moyenne (robustesse/operability) :

- D. Sauvegarder explicitement `defaults` (median dict) dans `processed_data.joblib`.
  - Implémentation : `processed_data['defaults'] = X_test_df.median().to_dict()` dans `save_processed_data()`.

- E. Limiter taille du `shap_b64` renvoyé et exécuter SHAP en background.

Priorité basse (hardening infra) :

- F. Dockerfile : ajouter `USER` non-root, ajouter healthcheck, pin deps.

---

4) Exemple de patch minimal recommandé (lazy-load)

Idée : dans `app/app.py`, remplacer l'appel immédiat `_load_resources()` par :

```
@app.on_event('startup')
def startup_event():
    try:
        _load_resources()
    except Exception as exc:
        logger.exception('Failed loading resources at startup')
        # depending on policy : raise to avoid starting with broken state

# Alternatively, use helper used by routes:
def _ensure_resources_loaded():
    if _model is None:
        _load_resources()

# call `_ensure_resources_loaded()` at beginning of routes that need model
```

Remarque : je peux générer le patch proposé et exécuter les tests locaux.

---

5) Annexes

- Voir fichiers de revue détaillés : `REVIEWS/SECURITY.md`, `REVIEWS/DATA_PIPELINE.md`, `REVIEWS/DEPLOYMENT_AND_CI.md`, `REVIEWS/MODEL_ARTIFACTS.md`, `REVIEWS/FRONTEND_AND_TEMPLATES.md`, `REVIEWS/TESTS_AND_CI.md`, `REVIEWS/NOTEBOOKS_AND_REPORTS.md`.

---

Si vous donnez le feu vert, je peux :
- appliquer le patch `lazy-load + logging + prevent-default-admin` et exécuter la suite de tests (ou seulement les tests pertinents),
- ou continuer l'audit en extrayant contenus binaires (inspection des artefacts modèles) si vous voulez que j'analyse métadonnées des fichiers `models/` (nécessite lecture binaire limitée — possible mais je ne peux pas exécuter du code arbitraire hors des fichiers de texte).
