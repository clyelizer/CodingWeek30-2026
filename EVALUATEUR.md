# EVALUATEUR — Évaluation stricte et exigeante

Date : 2026-03-15

Objet : documenter, noter et suivre l'avancement du projet « Pediatric appendicitis diagnosis ». Ce fichier sera maintenu à jour par l'évaluateur.

**Résumé**
- Etat général : Avancement solide — pipeline ML, API FastAPI, SHAP et tests unitaires présents.
- Qualité : Code structuré et professionnel. Points bloquants pour livraison : CI/CD absent, documentation de prompt engineering manquante, vérification de la présence du dataset brut nécessaire.

**Conformité au brief (statut)**
- `notebooks/` : présent — ouvrir `notebooks/eda.ipynb` pour vérifier réponses aux questions critiques (missing values, outliers, corrélations, balance). (À vérifier)
- `src/` : présent — contient `data_processing.py`, `train_model.py`, `shap_explanations.py`, `evaluate_model.py` (référencé). (OK)
- `app/` : présent — `app/app.py` (FastAPI) + templates. (OK)
- `tests/` : présent — tests unitaires pour API et utilitaires (`tests/test_app.py`, etc.). (OK)
- `optimize_memory(df)` : implémenté dans `src/data_processing.py` (✅). Ajouter test dédié. (OK → TEST)
- Modèles : `Random Forest` implémenté ; `LightGBM`/`CatBoost` supportés conditionnellement si installés. (OK)
- SHAP : implémenté dans `src/shap_explanations.py` avec fallback si `shap` absent. (OK)
- CI/CD : **ABSENT** — `.github/workflows` manquant. (BLOCKER)
- Prompt engineering docs : **ABSENT** — créer `PROMPTS.md` ou section `README.md`. (HIGH)
- Trello / gestion des tâches : non détecté in-repo. (OPTIONAL)

**Points forts**
- Pipeline de prétraitement complet, bonnes pratiques sklearn (pipelines, ColumnTransformer).
- `optimize_memory` pragmatique et bien conçu (downcast, catégories).
- SHAP intégré proprement et réutilisable.
- API FastAPI robuste (gestion modèle, sessions, historique SQLite).
- Tests unitaires couvrent de nombreux cas critiques.

**Faiblesses et risques (strict)**
1. Absence de CI/CD automatisée : sans CI la régression est probable et l'exigence du brief n'est pas satisfaite. (Critique)
2. Absence de documentation de prompt engineering : impossible d'évaluer la stratégie et les itérations d'IA. (Critique)
3. Incertitude sur la présence et la provenance du dataset brut (`data/raw/dataset.xlsx`) : reproductibilité compromise. (Élevé)
4. Tests manquants ciblant `optimize_memory` et vérifiant les gains mémoire/typage. (Moyen)
5. README.md à vérifier : doit contenir commandes exactes pour `pip install -r requirements.txt`, entraînement et exécution de l'app. (Moyen)

**Recommandations immédiates (priorité)**
- (P0) Ajouter workflow GitHub Actions (`.github/workflows/ci.yml`) : installe `requirements.txt` et exécute `pytest -q`. Bloquant avant merge final.
- (P0) Ajouter `PROMPTS.md` listant prompts utilisés (exactes invites, contexte, résultats et itérations). Exiger un exemple concret pour une tâche (ex : création `optimize_memory`).
- (P1) Vérifier / ajouter dataset brut ou script de récupération `scripts/download_data.py`; documenter origine et checksum.
- (P1) Ajouter test unitaire pour `optimize_memory(df)` et valider réduction mémoire et types.
- (P2) Compléter README.md avec instructions reproductibles et commande unique d'entraînement + lancement.

**Prochaine mise à jour prévue**
- J'actualiserai ce fichier après : ajout CI, ajout PROMPTS.md, et exécution des tests. Indiquez quand vous voulez que j'exécute ces modifications automatiquement.

-- Évaluateur (strict) — fichier maintenu par l'équipe d'évaluation
