# Sécurité — Revue détaillée

Fichier analysés principalement: `app/app.py`, `src/train_model.py`, `Dockerfile`, `requirements.txt`, `REVIEWS/PROJECT_REVIEW.md`.

Objectif: lister vecteurs d'attaque, mauvaises pratiques et recommandations
pragmatiques pour durcir l'application avant un déploiement réel.

---

1) Chargement d'artefacts Python (joblib)
- Constat : `joblib.load()` est utilisé sans validation forte dans `app/app.py`
  et `src/train_model.py` pour charger modèles et données sérialisées.
- Risque : les payloads joblib/pickle peuvent exécuter du code arbitraire
  lors du désérialisation si l'artefact est compromis. En production, cela
  ouvre une haute surface d'attaque (RCE via fichier modèle).
- Recommandations :
  - Ne jamais accepter des artefacts joblib/pickle depuis des sources
    non fiables. Documenter clairement les exigences (source, checksums).
  - Si possible, remplacer joblib/pickle par un format plus sûr (ONNX,
    PMML) pour les artefacts modèles, ou au moins valider la signature
    numérique (HMAC, GPG) avant `joblib.load()`.
  - Limiter les permissions des fichiers modèles (chmod 440) et stocker
    les artefacts dans un stockage contrôlé (S3 privé, registre d'artefacts).

2) Admin par défaut et mots de passe codés
- Constat : `_init_db()` dans `app/app.py` crée un utilisateur `admin`
  avec le mot de passe `admin123` si absent.
- Risque : déploiement accidentel en production avec accès admin connu.
- Recommandations :
  - Ne jamais créer de comptes par défaut en production. Si nécessaire,
    conditionner la création par une variable d'environnement explicite
    (p.ex. `CREATE_DEFAULT_ADMIN=true`) et loguer/alerter.
  - Exiger changement du mot de passe au premier usage ou stocker des
    secrets via un gestionnaire (Vault, AWS Secrets Manager).

3) Gestion des sessions et secret par défaut
- Constat : `_SECRET_KEY` par défaut `dev-secret-please-change` si
  `PEDIA_SECRET` non fourni. Le token de session est HMAC signé avec
  SHA256; cookie HTTPOnly est utilisé.
- Risque : clef faible par défaut — token manipulations / vulnérabilité
  si déployé sans configuration.
- Recommandations :
  - Forcer la présence d'une variable d'environnement secrète en CI/CD
    ou refuser le démarrage si `PEDIA_SECRET` est le placeholder.
  - Préférer des JWT signés ou sessions côté serveur (Redis) selon les
    exigences de scalabilité; documenter clairement la stratégie.

4) Bruteforce et protection du point de connexion
- Constat : endpoint `/login` accepte tentatives sans limit.
- Risque : bruteforce / credential stuffing.
- Recommandations :
  - Ajouter rate limiting (par IP / username) et verrouillage progressif.
  - Envisager CAPTCHA après N tentatives échouées.
  - Instrumenter logs d'authentification pour alertes SIEM.

5) Logs et affichage d'erreurs
- Constat : utilisation de `print()` pour erreurs non-fatal (SHAP, DB save).
- Risque : absence de niveaux log, difficulté d'agrégation en production.
- Recommandations :
  - Remplacer `print` par module `logging` configuré (handlers, rotation).
  - Ne pas exposer tracebacks dans les réponses HTTP; loguer stacktrace
    en backend avec corrélation d'ID de requête.

6) Protection des fichiers statiques / templates
- Constat : templates HTML incluent des chemins absolus `/static/...` et
  la page exporte PDF via `html2pdf` côté client.
- Risque : fuites d'informations si des ressources non protégées contiennent
  données sensibles. Le PDF peut inclure des données patients si non-filtré.
- Recommandations :
  - S'assurer que seules ressources statiques publiques sont dans `/static`.
  - Éviter l'inclusion de données sensibles non nécessaires dans la page
    ou fournir option d'anonymisation avant export PDF.

7) Hardening Docker
- Constat : Dockerfile installe toutes dépendances depuis `requirements.txt`
  et exécute `uvicorn` en tant que défaut CMD.
- Recommandations :
  - Utiliser un utilisateur non-root (`RUN groupadd -r app && useradd -r -g app app` et `USER app`).
  - Verrouiller versions de dépendances en `requirements.txt` (pin exact).
  - Scanner l'image construite pour vulnérabilités (trivy) en CI.

8) CI secrets et dépendances
- Constat : `ci.yml` installe dépendances et exécute tests; pas de secret
  exposé mais vérifier variables sensibles (Docker build pushing?).
- Recommandations :
  - Stocker `PEDIA_SECRET` comme secret GitHub et refuser build Docker si
    la variable n'est pas fournie pour release.
  - Utiliser cache pip et virtualenvs isolés pour reproducibility.

---

Priorité immédiate (3 actions):
1. Remplacer `print` par `logging` et vérifier logs.
2. Empêcher création automatique d'admin en production (env flag).
3. Interdire joblib.load sur fichiers non vérifiés; ajouter signature check.

Pour toute correction que vous voulez que je mette en patch, indiquez
les priorités et je préparerai PRs ciblées (ex: `app/app.py` safe-load).
