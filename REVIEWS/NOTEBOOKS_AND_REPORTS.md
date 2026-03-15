# Notebooks & Rapports — Revue détaillée

Fichiers analysés: `notebooks/eda.ipynb`, `notebooks/eda-etude_correlation.ipynb`, `reports/`.

Objectif: vérifier reproducibility, sensibilité des notebooks, et bonnes pratiques
pour publication et archivage des résultats.

---

1) Notebooks — observations
- Les notebooks contiennent EDA et analyses corrélations. Les notebooks sont
  utiles pour exploration mais posent des risques de reproductibilité :
  - sorties (plots, tables) souvent incluses dans le notebook, rendant la
    revue difficile si les données changent.
  - cellules contenant code data-specific (chemins absolus) peuvent casser
    sur une autre machine.

Recommandations notebooks :
- Convertir analyses reproductibles en scripts (`notebooks/` → `reports/scripts`)
  ou utiliser `nbconvert` avec `papermill` pour exécution automatisée.
- Nettoyer ou supprimer données sensibles dans notebooks avant commit.
- Ajouter `requirements-notebook.txt` ou binder/conda env yml pour faciliter
  reproduction.

2) Rapports et figures
- Observé : `reports/` est présent mais peu structuré. Les figures SHAP et
  courbes ROC sont générées par `src/train_model.py` et `src/evaluate_model.py`.

Recommandations rapports :
- Standardiser emplacement `reports/figures/YYYY-MM-DD/` pour chaque run.
- Versionner rapports clefs (ROC, SHAP) séparément ou stocker dans CI artifacts.

3) Sensibilité clinique
- Rappel : toute visualisation ou export (PDF) doit inclure avertissement
  médical (déjà présent dans README) et datas anonymisées si rapport
  partagés.

---

Priorité immédiate (2 actions):
1. Ajouter un petit README dans `notebooks/` décrivant comment exécuter
   et reproduire les analyses (papermill / env).
2. Ajouter script `scripts/generate_reports.sh` qui exécute notebooks et stocke
   figures dans un dossier daté.
