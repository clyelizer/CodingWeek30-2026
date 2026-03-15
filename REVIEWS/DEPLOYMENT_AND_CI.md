# Déploiement & CI — Revue détaillée

Fichiers analysés: `Dockerfile`, `.github/workflows/ci.yml`, `requirements.txt`, `REVIEWS/SECURITY.md`.

Objectif: vérifier robustesse de la chaîne de build/test/déploiement,
fiabilité des images, recommendations pour production.

---

1) CI workflow — observations
- Le workflow contient deux définitions : une ancienne et une plus récente
  (doublon dans le fichier). Consolidation requise pour éviter confusions.
  - Première partie cible Python 3.10 et exécute tests basiques.
  - Suite plus récente matrix Python 3.11 exécute tests avec coverage et
    build Docker sur `main`.
- Risque : logique du CI contradictoire ou actions non exécutées à cause
  du double `name: CI` et sections dupliquées.

Recommandations CI :
- Nettoyer le workflow pour n'avoir qu'une seule définition claire, idéalement
  matrix [3.10, 3.11] si compatibilité voulue, ou 3.11 seulement si ciblé.
- Activer caches (déjà présent) et fixtures pour accélérer.
- Ajouter étape statique : `pip-audit` ou `safety` pour vérifier vulnérabilités
  dans `requirements.txt` avant install.
- Ajouter step `docker scan` (trivy) dans le job `docker` pour détection vulnérabilités.

2) Dockerfile — observations
- Image basée sur `python:3.11-slim` (bon choix taille/compatibilité).
- Copie `models/` et `data/processed/` dans l'image ; cela est pratique
  pour conteneur standalone mais augmente la taille image.
- CMD démarre `uvicorn app.app:app` en production — utile mais manque
  paramètres `--proxy-headers` et `--loop uvloop`/workers recommandés.

Recommandations Docker :
- Construire image multi-stage si des étapes build lourdes sont requises.
- Exécuter un utilisateur non-root dans l'image (USER non-root).
- Pour uvicorn en production : utiliser `gunicorn` + `uvicorn.workers.UvicornWorker`
  ou `uvicorn --workers 4` selon CPU; ajouter `--proxy-headers` pour run derrière
  reverse-proxy.
- Minimiser layers en combinant `COPY` et `RUN` où pertinent et nettoyer cache.

3) Reproducibilité & dépendances
- `requirements.txt` contient versions larges (e.g., fastapi>=0.118,<0.136).
  Cela est pratique mais peut engendrer différences entre environnements.

Recommandations :
- Pour images de production, utiliser des `requirements.txt` pinnings exacts
  générés via `pip freeze` depuis un environnement contrôlé.
- Documenter la version cible Python (fichier `.python-version` existe).

4) Secrets et variables d'environnement
- Garantir que `PEDIA_SECRET` soit fourni via CI/CD secrets et ne soit pas
  commité.
- Ajouter vérification au démarrage de l'app (fail fast) si des secrets
  nécessaires ne sont pas fournis (sauf en dev).

5) Smoke tests & healthchecks
- Le workflow `docker` fait un `curl /` comme smoke-test; recommander :
  - attendre activement jusqu'à readiness (retry with timeout) plutôt
    que sleep 10.
  - exposer endpoint `/health` renvoyant status détaillé (ok database, model loaded).

6) Images & taille
- Copier `models/` et `data/processed/` peut gonfler image; envisager
  d'utiliser volumes ou fetch au runtime depuis stockage privé pour
  diminuer taille d'image (et permettre mises à jour de modèles sans
  rebuild complet).

---

Priorité immédiate (3 actions):
1. Corriger le workflow CI pour supprimer duplication et standardiser matrix.
2. Ajouter `pips-audit`/`safety` scanning étape dans CI.
3. Hardener Dockerfile: non-root user + uvicorn production flags + healthcheck.
