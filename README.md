# PediAppendix — Aide au diagnostic pédiatrique de l'appendicite

## Présentation

**PediAppendix** est un système d'aide à la décision clinique pour le diagnostic
de l'appendicite pédiatrique.

À partir de **10 paramètres cliniques courants** (examen physique, biologie,
échographie), il prédit la probabilité d'appendicite et fournit une explication
SHAP détaillée de chaque prédiction.

**Dataset :** Regensburg Pediatric Appendicitis (UCI), n = 776 patients.  
**Modèle :** Random Forest — AUC-ROC = **0.9287** sur le jeu de test (n = 156).

---

## Architecture du projet

```
projet/
├── data/
│   ├── raw/          data_finale.xlsx         (776 patients, 27 variables)
│   └── processed/    processed_data.joblib    (split train/test stratifié 80/20)
├── models/
│   ├── random_forest.joblib        ← modèle de production (AUC 0.9287)
│   ├── gradient_boosting.joblib    (AUC 0.9141)
│   ├── logistic_regression.joblib  (AUC 0.8283)
│   └── svm.joblib                  (AUC 0.8102)
├── src/
│   ├── data_processing.py   pipeline de traitement des données
│   ├── train_model.py       entraînement et évaluation des modèles
│   └── evaluate_model.py    prédiction individuelle + explications SHAP
├── tests/
│   ├── test_data_processing.py   11 tests
│   ├── test_model.py             15 tests
│   └── test_evaluate_model.py    8 tests        → 34 tests total, tous passent
├── app/
│   ├── app.py               interface FastAPI
│   └── templates/
│       ├── index.html       formulaire de saisie (10 features)
│       └── result.html      page de résultat + graphique SHAP
├── notebooks/
│   └── eda.ipynb            analyse exploratoire du dataset
├── MD/
│   ├── README.md            index de la documentation
│   ├── 01_data_processing.md
│   ├── 02_train_model.md
│   ├── 03_evaluate_model.md
│   └── 04_webapp.md
├── conftest.py              configuration pytest (sys.path)
├── requirements.txt
└── Dockerfile
```

---

# 📊 Organisation & Gestion de Projet

Le projet a été géré via **Jira Atlassian**. Voici la répartition des tâches :

### 👤 Rôle : teamlead
<details>
<summary>Cliquez pour voir les tâches de teamlead</summary>

| Issue Type | Key | Summary | Assignee | Reporter | Priority | Status | Resolution | Created | Updated | Due date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Task | [SUP-40](https://grp31.atlassian.net/browse/SUP-40) | TeamLead-Valider la reproductibilité du projet. ‎ | coulibaly ELISE | Diallo Nassirou Amadou Oumar | Medium | Open | Unresolved | 14/Mar/26 10:32 PM | 14/Mar/26 10:32 PM | &nbsp; |
| Task | [SUP-39](https://grp31.atlassian.net/browse/SUP-39) | TeamLead-Créer un Dockerfile pour conteneuriser l’application. | coulibaly ELISE | Diallo Nassirou Amadou Oumar | Medium | Open | Unresolved | 14/Mar/26 10:32 PM | 14/Mar/26 10:32 PM | &nbsp; |
| Task | [SUP-38](https://grp31.atlassian.net/browse/SUP-38) | TeamLead- Finaliser le README avec toutes les réponses aux questions critiques. | coulibaly ELISE | Diallo Nassirou Amadou Oumar | Medium | Open | Unresolved | 14/Mar/26 10:31 PM | 14/Mar/26 10:31 PM | &nbsp; |
| Task | [SUP-37](https://grp31.atlassian.net/browse/SUP-37) | TeamLead- Documenter l’ingénierie des prompts pour une tâche spécifique. | coulibaly ELISE | Diallo Nassirou Amadou Oumar | Medium | Open | Unresolved | 14/Mar/26 10:31 PM | 14/Mar/26 10:35 PM | &nbsp; |
| Task | [SUP-36](https://grp31.atlassian.net/browse/SUP-36) | TeamLead-Coordonner l’intégration des différentes branches. | coulibaly ELISE | Diallo Nassirou Amadou Oumar | Medium | Open | Unresolved | 14/Mar/26 10:30 PM | 14/Mar/26 10:30 PM | &nbsp; |
| Task | [SUP-35](https://grp31.atlassian.net/browse/SUP-35) | TeamLead- Mettre en place GitHub Actions (.github/workflows/ci.yml) avec un test minimal. | coulibaly ELISE | Diallo Nassirou Amadou Oumar | Medium | Open | Unresolved | 14/Mar/26 10:29 PM | 14/Mar/26 10:29 PM | &nbsp; |
| Task | [SUP-34](https://grp31.atlassian.net/browse/SUP-34) | TeamLead-Initialiser le README.md avec la description du projet. | coulibaly ELISE | Diallo Nassirou Amadou Oumar | Medium | Open | Unresolved | 14/Mar/26 10:28 PM | 14/Mar/26 10:28 PM | &nbsp; |
| Task | [SUP-33](https://grp31.atlassian.net/browse/SUP-33) | TeamLead-Configurer le tableau jira et inviter l’équipe | coulibaly ELISE | Diallo Nassirou Amadou Oumar | Medium | Open | Unresolved | 14/Mar/26 10:27 PM | 14/Mar/26 10:27 PM | &nbsp; |
| Task | [SUP-32](https://grp31.atlassian.net/browse/SUP-32) | TeamLead-Configurer le tableau jira et inviter l’équipe | coulibaly ELISE | Diallo Nassirou Amadou Oumar | Medium | Open | Unresolved | 14/Mar/26 10:26 PM | 14/Mar/26 10:26 PM | &nbsp; |
| Task | [SUP-31](https://grp31.atlassian.net/browse/SUP-31) | TeamLead-Créer le dépôt GitHub et la structure de dossiers | coulibaly ELISE | Diallo Nassirou Amadou Oumar | Medium | Open | Unresolved | 14/Mar/26 10:25 PM | 14/Mar/26 10:25 PM | &nbsp; |

</details>

### 👤 Rôle : dataEngineer
<details>
<summary>Cliquez pour voir les tâches de dataEngineer</summary>

| Issue Type | Key | Summary | Assignee | Reporter | Priority | Status | Resolution | Created | Updated | Due date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Task | [SUP-23](https://grp31.atlassian.net/browse/SUP-23) | Data Engineering_Documenter les fonctions dans le code (docstrings). | Diallo Nassirou Amadou Oumar | Diallo Nassirou Amadou Oumar | Medium | Open | Unresolved | 14/Mar/26 10:12 PM | 14/Mar/26 10:12 PM | &nbsp; |
| Task | [SUP-22](https://grp31.atlassian.net/browse/SUP-22) | Data Engineering_ Écrire les tests unitaires dans tests/test_data_processing.py. | Diallo Nassirou Amadou Oumar | Diallo Nassirou Amadou Oumar | Medium | Open | Unresolved | 14/Mar/26 10:10 PM | 14/Mar/26 10:10 PM | &nbsp; |
| Task | [SUP-21](https://grp31.atlassian.net/browse/SUP-21) | Data Engineering_ Participer à la rédaction des sections README concernant les données | Diallo Nassirou Amadou Oumar | Diallo Nassirou Amadou Oumar | Medium | Open | Unresolved | 14/Mar/26 10:05 PM | 14/Mar/26 10:05 PM | &nbsp; |
| Task | [SUP-20](https://grp31.atlassian.net/browse/SUP-20) | Data Engineering - Creer pipeline de pretraitement complet (NA, encodage, normalisation) | Diallo Nassirou Amadou Oumar | coulibaly ELISE | Medium | Open | Unresolved | 14/Mar/26 6:30 PM | 14/Mar/26 6:30 PM | &nbsp; |
| Task | [SUP-19](https://grp31.atlassian.net/browse/SUP-19) | Data Engineering - Implementer optimize_memory(df) dans src/data_processing.py | Diallo Nassirou Amadou Oumar | coulibaly ELISE | Medium | Open | Unresolved | 14/Mar/26 6:22 PM | 14/Mar/26 6:22 PM | &nbsp; |
| Task | [SUP-18](https://grp31.atlassian.net/browse/SUP-18) | Data Engineering - Fournir un resume clair des conclusions a l equipe | Diallo Nassirou Amadou Oumar | coulibaly ELISE | Medium | Open | Unresolved | 14/Mar/26 6:20 PM | 14/Mar/26 6:20 PM | &nbsp; |
| Task | [SUP-17](https://grp31.atlassian.net/browse/SUP-17) | Data Engineering - Calculer matrice de correlation et identifier features importantes | Diallo Nassirou Amadou Oumar | coulibaly ELISE | Medium | Open | Unresolved | 14/Mar/26 6:18 PM | 14/Mar/26 6:18 PM | &nbsp; |

</details>

### 👤 Rôle : dataAnalyst
<details>
<summary>Cliquez pour voir les tâches de dataAnalyst</summary>

| Issue Type | Key | Summary | Assignee | Reporter | Priority | Status | Resolution | Created | Updated | Due date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Task | [SUP-10](https://grp31.atlassian.net/browse/SUP-10) | EDA - Verification de l&#39;equilibre des classes | Ange Sarah | coulibaly ELISE | Medium | Open | Unresolved | 14/Mar/26 6:07 PM | 14/Mar/26 6:07 PM | &nbsp; |
| Task | [SUP-9](https://grp31.atlassian.net/browse/SUP-9) | EDA - Détection et traitement des outliers | Ange Sarah | coulibaly ELISE | Medium | Open | Unresolved | 14/Mar/26 6:07 PM | 14/Mar/26 6:07 PM | &nbsp; |
| Task | [SUP-8](https://grp31.atlassian.net/browse/SUP-8) | EDA - Analyse des valeurs manquantes | Ange Sarah | coulibaly ELISE | Medium | Open | Unresolved | 14/Mar/26 6:07 PM | 14/Mar/26 6:07 PM | &nbsp; |

</details>

### 👤 Rôle : iaEngineer
<details>
<summary>Cliquez pour voir les tâches de iaEngineer</summary>

| Issue Type | Key | Summary | Assignee | Reporter | Priority | Status | Resolution | Created | Updated | Due date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Task | [SUP-30](https://grp31.atlassian.net/browse/SUP-30) | ML-Engineer-Fournir à AD les informations nécessaires pour l’intégration. | MOHAMED JOUAHAR | Diallo Nassirou Amadou Oumar | Medium | Open | Unresolved | 14/Mar/26 10:20 PM | 14/Mar/26 10:20 PM | &nbsp; |
| Task | [SUP-29](https://grp31.atlassian.net/browse/SUP-29) | ML-Engineer- Écrire les tests dans tests/test_model.py. | MOHAMED JOUAHAR | Diallo Nassirou Amadou Oumar | Medium | Open | Unresolved | 14/Mar/26 10:19 PM | 14/Mar/26 10:19 PM | &nbsp; |
| Task | [SUP-28](https://grp31.atlassian.net/browse/SUP-28) | ML-Engineer- Intégrer SHAP (valeurs, graphiques : summary plot, dependance plot, force plot). | MOHAMED JOUAHAR | Diallo Nassirou Amadou Oumar | Medium | Open | Unresolved | 14/Mar/26 10:18 PM | 14/Mar/26 10:18 PM | &nbsp; |
| Task | [SUP-27](https://grp31.atlassian.net/browse/SUP-27) | ML-Engineer- Sauvegarder le modèle final (apk) et le préprocesseur associé.. | MOHAMED JOUAHAR | Diallo Nassirou Amadou Oumar | Medium | Open | Unresolved | 14/Mar/26 10:17 PM | 14/Mar/26 10:17 PM | &nbsp; |
| Task | [SUP-26](https://grp31.atlassian.net/browse/SUP-26) | ML-Engineer-Comparer les performances et sélectionner le meilleur modèle. | MOHAMED JOUAHAR | Diallo Nassirou Amadou Oumar | Medium | Open | Unresolved | 14/Mar/26 10:16 PM | 14/Mar/26 10:16 PM | &nbsp; |
| Task | [SUP-25](https://grp31.atlassian.net/browse/SUP-25) | ML-Engineer- Implémenter l’entraînement et l’évaluation avec validation croisée. | MOHAMED JOUAHAR | Diallo Nassirou Amadou Oumar | Medium | Open | Unresolved | 14/Mar/26 10:15 PM | 14/Mar/26 10:15 PM | &nbsp; |
| Task | [SUP-24](https://grp31.atlassian.net/browse/SUP-24) | ML-Engineer- Entraîner au moins trois modèles (SVM, Random Forest, LightGBM, CatBoost | MOHAMED JOUAHAR | Diallo Nassirou Amadou Oumar | Medium | Open | Unresolved | 14/Mar/26 10:14 PM | 14/Mar/26 10:14 PM | &nbsp; |

</details>

### 👤 Rôle : webdev
<details>
<summary>Cliquez pour voir les tâches de webdev</summary>

| Issue Type | Key | Summary | Assignee | Reporter | Priority | Status | Resolution | Created | Updated | Due date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Task | [SUP-16](https://grp31.atlassian.net/browse/SUP-16) | Streamlit App - Tester l’application manuellement | Sanogo | coulibaly ELISE | Medium | Open | Unresolved | 14/Mar/26 6:17 PM | 14/Mar/26 6:17 PM | &nbsp; |
| Task | [SUP-15](https://grp31.atlassian.net/browse/SUP-15) | Streamlit App - Intégrer visualisations SHAP | Sanogo | coulibaly ELISE | Medium | Open | Unresolved | 14/Mar/26 6:17 PM | 14/Mar/26 6:17 PM | &nbsp; |
| Task | [SUP-14](https://grp31.atlassian.net/browse/SUP-14) | Streamlit App - Afficher probabilité et classe prédite | Sanogo | coulibaly ELISE | Medium | Open | Unresolved | 14/Mar/26 6:17 PM | 14/Mar/26 6:17 PM | &nbsp; |
| Task | [SUP-13](https://grp31.atlassian.net/browse/SUP-13) | Streamlit App - Charger modèle et préprocesseur | Sanogo | coulibaly ELISE | Medium | Open | Unresolved | 14/Mar/26 6:11 PM | 14/Mar/26 6:11 PM | &nbsp; |
| Task | [SUP-12](https://grp31.atlassian.net/browse/SUP-12) | Streamlit App - Concevoir interface utilisateur | Sanogo | coulibaly ELISE | Medium | Open | Unresolved | 14/Mar/26 6:11 PM | 14/Mar/26 6:11 PM | &nbsp; |
| Task | [SUP-11](https://grp31.atlassian.net/browse/SUP-11) | Streamlit App - Develop main application (app/app.py) | Sanogo | coulibaly ELISE | Medium | Open | Unresolved | 14/Mar/26 6:09 PM | 14/Mar/26 6:09 PM | &nbsp; |

</details>



---

## Features du modèle (10)

| # | Variable | Type | Source clinique |
|---|----------|------|----------------|
| 1 | `Lower_Right_Abd_Pain` | Binaire (oui/non) | Examen clinique |
| 2 | `Migratory_Pain` | Binaire (oui/non) | Examen clinique |
| 3 | `Ipsilateral_Rebound_Tenderness` | Binaire (oui/non) | Examen clinique |
| 4 | `Nausea` | Binaire (oui/non) | Examen clinique |
| 5 | `Body_Temperature` | Numérique (°C) | Examen clinique |
| 6 | `WBC_Count` | Numérique (G/L) | Biologie |
| 7 | `Neutrophil_Percentage` | Numérique (%) | Biologie |
| 8 | `CRP` | Numérique (mg/L) | Biologie |
| 9 | `Appendix_Diameter` | Numérique (mm) | Échographie |
| 10 | `Age` | Numérique (années) | Démographique |

---

## Résultats

| Modèle | AUC-ROC | F1 (macro) | Accuracy |
|--------|---------|------------|----------|
| **Random Forest** ← retenu | **0.9287** | **0.8457** | **0.8526** |
| Gradient Boosting | 0.9141 | 0.8178 | 0.8269 |
| Logistic Regression | 0.8283 | 0.7354 | 0.7564 |
| SVM (RBF) | 0.8102 | 0.7198 | 0.7436 |

**Interprétation de l'AUC = 0.9287 :** en tirant aléatoirement un patient positif
et un patient négatif, le modèle attribue une probabilité plus élevée au positif
dans 92.87% des cas.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Utilisation

### 1. Traitement des données (une seule fois)
```bash
python src/data_processing.py
# → data/processed/processed_data.joblib
```

### 2. Entraînement des modèles (une seule fois)
```bash
python src/train_model.py
# → models/random_forest.joblib (+ 3 autres modèles)
```

### 3. Tests unitaires
```bash
python -m pytest tests/ --rootdir="." -q
# 34 passed
```

### 4. Lancement de l'application web
```bash
uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
# → http://localhost:8000
```

### Docker
```bash
docker build -t pediappendix .
docker run -p 8000:8000 pediappendix
```

---

## Paradigme de développement

Ce projet suit un paradigme **fonctionnel strict** :
- **Une fonction = une tâche précise et testable**
- **Un test = une fonction = une assertion**
- Pas d'état global mutable entre fonctions
- Pas de data leakage : StandardScaler encapsulé dans chaque Pipeline sklearn

---

## Documentation technique

Voir le dossier [`MD/`](MD/README.md) pour la documentation détaillée
de chaque module avec les sorties et décisions de conception.

---

> ⚠️ **Avertissement médical** — Cet outil est à usage expérimental uniquement.
> Il ne remplace pas le jugement clinique d'un professionnel de santé.
> Dataset : Regensburg Pediatric Appendicitis — UCI Machine Learning Repository.
