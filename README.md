# PediAppendix — Aide au diagnostic pédiatrique de l'appendicite

## Présentation

**PediAppendix** est un système d'aide à la décision clinique pour le diagnostic de l'appendicite pédiatrique. À partir de **10 paramètres cliniques courants** (examen physique, biologie, échographie), il prédit la probabilité d'appendicite et fournit une explication SHAP détaillée.

### 👥 Équipe Projet — GROUPE 31
| Membre | Rôle |
|--------|------|
| **Coulibaly ELISE** | Teamlead & Coordination |
| **Diallo Nassirou Amadou Oumar** | Data Engineer & Pipeline |
| **MOHAMED JOUAHAR** | IA Engineer & Modélisation |
| **Sanogo** | Web Developer (Streamlit) |
| **Ange Sarah** | Data Analyst (EDA) |

**Dataset :** Regensburg Pediatric Appendicitis (UCI), n = 776 patients.  
**Modèle retenu :** Random Forest — AUC-ROC = **0.9359** (Best model).

### 📈 Résultats et Interprétabilité

| Courbe ROC (Performance) | Importance des Features (SHAP) |
|:---:|:---:|
| ![ROC Curve](file:///c:/Users/clyel/Desktop/new/CodingWeek30-2026/reports/figures/roc_Random_Forest.png) | ![SHAP Summary](file:///c:/Users/clyel/Desktop/new/CodingWeek30-2026/reports/figures/shap_summary.png) |

> [!NOTE]
> Le modèle Random Forest a été sélectionné pour sa stabilité en validation croisée (moyenne 0.92) et son excellente capacité de généralisation sur le jeu de test.

---


## Présentation du Groupe projet 
 GROUPE 31 

Coulibaly eli... (Teamlead)
....
.....

---
## Architecture du projet

```
projet/
├── data/
│   ├── raw/          dataset.xlsx             (776 patients, 27 variables)
│   └── processed/    processed_data.joblib    (split train/test stratifié 80/20)
├── models/
│   ├── preprocessor.pkl           ← préprocesseur (StandardScaler)
│   ├── random_forest_model.pkl    ← modèle de production (AUC ~0.92)
│   └── ...
├── src/
│   ├── data_processing.py   pipeline de traitement des données
│   ├── train_model.py       entraînement et évaluation des modèles
│   └── evaluate_model.py    prédiction individuelle + explications SHAP
├── tests/
│   ├── test_data_processing.py   11 tests
│   └── test_model.py             2 tests        → 13 tests total, tous passent
├── app/
│   ├── app.py               interface FastAPI
│   ├── static/              CSS et assets
│   └── templates/
│       ├── landing_page.html
│       ├── auth.html
│       ├── diagnosis_form.html
│       └── diagnosis_result.html
├── notebooks/
│   └── eda.ipynb            analyse exploratoire du dataset
├── MD/                      Documentation technique détaillée
├── requirements.txt
└── Dockerfile
```

---

# 📊 Organisation & Gestion de Projet

Le projet a été géré via **Jira Atlassian**. 
Nous avons opte pour un partage des tahces par role , selon les 5 roles repartis  
Voici la répartition des tâches :

### 👤 Rôle : teamlead
<details>
<summary>Cliquez pour voir les tâches de teamlead</summary>

| Issue Type | Key | Summary | Assignee | Reporter | Priority | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Task | [SUP-40](https://grp31.atlassian.net/browse/SUP-40) | TeamLead-Valider la reproductibilité du projet. ‎ | coulibaly ELISE | Diallo Nassirou Amadou Oumar | Medium | Open |
| Task | [SUP-39](https://grp31.atlassian.net/browse/SUP-39) | TeamLead-Créer un Dockerfile pour conteneuriser l’application. | coulibaly ELISE | Diallo Nassirou Amadou Oumar | Medium | Open |
| Task | [SUP-38](https://grp31.atlassian.net/browse/SUP-38) | TeamLead- Finaliser le README avec toutes les réponses aux questions critiques. | coulibaly ELISE | Diallo Nassirou Amadou Oumar | Medium | Open |
| Task | [SUP-37](https://grp31.atlassian.net/browse/SUP-37) | TeamLead- Documenter l’ingénierie des prompts pour une tâche spécifique. | coulibaly ELISE | Diallo Nassirou Amadou Oumar | Medium | Open |
| Task | [SUP-36](https://grp31.atlassian.net/browse/SUP-36) | TeamLead-Coordonner l’intégration des différentes branches. | coulibaly ELISE | Diallo Nassirou Amadou Oumar | Medium | Open |
| Task | [SUP-35](https://grp31.atlassian.net/browse/SUP-35) | TeamLead- Mettre en place GitHub Actions (.github/workflows/ci.yml) avec un test minimal. | coulibaly ELISE | Diallo Nassirou Amadou Oumar | Medium | Open |
| Task | [SUP-34](https://grp31.atlassian.net/browse/SUP-34) | TeamLead-Initialiser le README.md avec la description du projet. | coulibaly ELISE | Diallo Nassirou Amadou Oumar | Medium | Open |
| Task | [SUP-33](https://grp31.atlassian.net/browse/SUP-33) | TeamLead-Configurer le tableau jira et inviter l’équipe | coulibaly ELISE | Diallo Nassirou Amadou Oumar | Medium | Open |
| Task | [SUP-32](https://grp31.atlassian.net/browse/SUP-32) | TeamLead-Configurer le tableau jira et inviter l’équipe | coulibaly ELISE | Diallo Nassirou Amadou Oumar | Medium | Open |
| Task | [SUP-31](https://grp31.atlassian.net/browse/SUP-31) | TeamLead-Créer le dépôt GitHub et la structure de dossiers | coulibaly ELISE | Diallo Nassirou Amadou Oumar | Medium | Open |

</details>

### 👤 Rôle : dataEngineer
<details>
<summary>Cliquez pour voir les tâches de dataEngineer</summary>

| Issue Type | Key | Summary | Assignee | Reporter | Priority | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Task | [SUP-23](https://grp31.atlassian.net/browse/SUP-23) | Data Engineering_Documenter les fonctions dans le code (docstrings). | Diallo Nassirou Amadou Oumar | Diallo Nassirou Amadou Oumar | Medium | Open |
| Task | [SUP-22](https://grp31.atlassian.net/browse/SUP-22) | Data Engineering_ Écrire les tests unitaires dans tests/test_data_processing.py. | Diallo Nassirou Amadou Oumar | Diallo Nassirou Amadou Oumar | Medium | Open |
| Task | [SUP-21](https://grp31.atlassian.net/browse/SUP-21) | Data Engineering_ Participer à la rédaction des sections README concernant les données | Diallo Nassirou Amadou Oumar | Diallo Nassirou Amadou Oumar | Medium | Open |
| Task | [SUP-20](https://grp31.atlassian.net/browse/SUP-20) | Data Engineering - Creer pipeline de pretraitement complet (NA, encodage, normalisation) | Diallo Nassirou Amadou Oumar | coulibaly ELISE | Medium | Open |
| Task | [SUP-19](https://grp31.atlassian.net/browse/SUP-19) | Data Engineering - Implementer optimize_memory(df) dans src/data_processing.py | Diallo Nassirou Amadou Oumar | coulibaly ELISE | Medium | Open |
| Task | [SUP-18](https://grp31.atlassian.net/browse/SUP-18) | Data Engineering - Fournir un resume clair des conclusions a l equipe | Diallo Nassirou Amadou Oumar | coulibaly ELISE | Medium | Open |
| Task | [SUP-17](https://grp31.atlassian.net/browse/SUP-17) | Data Engineering - Calculer matrice de correlation et identifier features importantes | Diallo Nassirou Amadou Oumar | coulibaly ELISE | Medium | Open |

</details>

### 👤 Rôle : dataAnalyst
<details>
<summary>Cliquez pour voir les tâches de dataAnalyst</summary>

| Issue Type | Key | Summary | Assignee | Reporter | Priority | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Task | [SUP-10](https://grp31.atlassian.net/browse/SUP-10) | EDA - Verification de l&#39;equilibre des classes | Ange Sarah | coulibaly ELISE | Medium | Open |
| Task | [SUP-9](https://grp31.atlassian.net/browse/SUP-9) | EDA - Détection et traitement des outliers | Ange Sarah | coulibaly ELISE | Medium | Open |
| Task | [SUP-8](https://grp31.atlassian.net/browse/SUP-8) | EDA - Analyse des valeurs manquantes | Ange Sarah | coulibaly ELISE | Medium | Open |

</details>

### 👤 Rôle : iaEngineer
<details>
<summary>Cliquez pour voir les tâches de iaEngineer</summary>

| Issue Type | Key | Summary | Assignee | Reporter | Priority | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Task | [SUP-30](https://grp31.atlassian.net/browse/SUP-30) | ML-Engineer-Fournir à AD les informations nécessaires pour l’intégration. | MOHAMED JOUAHAR | Diallo Nassirou Amadou Oumar | Medium | Open |
| Task | [SUP-29](https://grp31.atlassian.net/browse/SUP-29) | ML-Engineer- Écrire les tests dans tests/test_model.py. | MOHAMED JOUAHAR | Diallo Nassirou Amadou Oumar | Medium | Open |
| Task | [SUP-28](https://grp31.atlassian.net/browse/SUP-28) | ML-Engineer- Intégrer SHAP (valeurs, graphiques : summary plot, dependance plot, force plot). | MOHAMED JOUAHAR | Diallo Nassirou Amadou Oumar | Medium | Open |
| Task | [SUP-27](https://grp31.atlassian.net/browse/SUP-27) | ML-Engineer- Sauvegarder le modèle final (apk) et le préprocesseur associé.. | MOHAMED JOUAHAR | Diallo Nassirou Amadou Oumar | Medium | Open |
| Task | [SUP-26](https://grp31.atlassian.net/browse/SUP-26) | ML-Engineer-Comparer les performances et sélectionner le meilleur modèle. | MOHAMED JOUAHAR | Diallo Nassirou Amadou Oumar | Medium | Open |
| Task | [SUP-25](https://grp31.atlassian.net/browse/SUP-25) | ML-Engineer- Implémenter l’entraînement et l’évaluation avec validation croisée. | MOHAMED JOUAHAR | Diallo Nassirou Amadou Oumar | Medium | Open |
| Task | [SUP-24](https://grp31.atlassian.net/browse/SUP-24) | ML-Engineer- Entraîner au moins trois modèles (SVM, Random Forest, LightGBM, CatBoost | MOHAMED JOUAHAR | Diallo Nassirou Amadou Oumar | Medium | Open |

</details>

### 👤 Rôle : webdev
<details>
<summary>Cliquez pour voir les tâches de webdev</summary>

| Issue Type | Key | Summary | Assignee | Reporter | Priority | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Task | [SUP-16](https://grp31.atlassian.net/browse/SUP-16) | Streamlit App - Tester l’application manuellement | Sanogo | coulibaly ELISE | Medium | Open |
| Task | [SUP-15](https://grp31.atlassian.net/browse/SUP-15) | Streamlit App - Intégrer visualisations SHAP | Sanogo | coulibaly ELISE | Medium | Open |
| Task | [SUP-14](https://grp31.atlassian.net/browse/SUP-14) | Streamlit App - Afficher probabilité et classe prédite | Sanogo | coulibaly ELISE | Medium | Open |
| Task | [SUP-13](https://grp31.atlassian.net/browse/SUP-13) | Streamlit App - Charger modèle et préprocesseur | Sanogo | coulibaly ELISE | Medium | Open |
| Task | [SUP-12](https://grp31.atlassian.net/browse/SUP-12) | Streamlit App - Concevoir interface utilisateur | Sanogo | coulibaly ELISE | Medium | Open |
| Task | [SUP-11](https://grp31.atlassian.net/browse/SUP-11) | Streamlit App - Develop main application (app/app.py) | Sanogo | coulibaly ELISE | Medium | Open |

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
# → models/random_forest_model.pkl
```

### 3. Tests unitaires
```bash
python -m pytest tests/ --rootdir="." -q
# 13 passed
```

### 4. Lancement de l'application web
```bash
streamlit run app/app.py
# → http://localhost:8501
```

### 🧠 Prompt Engineering Task
L'IA a été utilisée pour :
1. **Optimisation** : Génération de la fonction `optimize_memory` pour réduire l'empreinte mémoire de 18%.
2. **Refactoring** : Migration de l'interface FastAPI vers Streamlit pour une intégration native des graphiques SHAP.
3. **Robustesse** : Création de tests unitaires dynamiques s'adaptant aux colonnes du préprocesseur.

### Docker
```bash
docker build -t pediappendix .
docker run -p 8000:8000 pediappendix
```

---

## Paradigme de développement

Ce projet suit un paradigme **fonctionnel ** :
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
