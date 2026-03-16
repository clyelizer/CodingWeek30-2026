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

Le projet a été géré via **Jira Atlassian**, avec une répartition claire des rôles et des responsabilités. Voici les extraits des tableaux de bord de l'équipe :

### 👤 Rôle : teamlead
<details>
<summary>Cliquez pour voir les tâches de teamlead</summary>

<table border="1">

                        <thead>
        <tr class="rowHeader">
            
                                                            <th class="colHeaderLink headerrow-issuetype" data-id="issuetype">
                                                        Issue Type
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-issuekey" data-id="issuekey">
                                                        Key
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-summary" data-id="summary">
                                                        Summary
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-assignee" data-id="assignee">
                                                        Assignee
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-reporter" data-id="reporter">
                                                        Reporter
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-priority" data-id="priority">
                                                        Priority
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-status" data-id="status">
                                                        Status
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-resolution" data-id="resolution">
                                                        Resolution
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-created" data-id="created">
                                                        Created
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-updated" data-id="updated">
                                                        Updated
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-duedate" data-id="duedate">
                                                        Due date
                                                    </th>
                                                                    </tr>
    </thead>
    <tbody>
                    

                <tr id="issuerow10115" rel="10115" data-issuekey="SUP-40" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-40" href="https://grp31.atlassian.net/browse/SUP-40">SUP-40</a>
</td>
                                            <td class="summary"><p>
                TeamLead-Valider la reproductibilité du projet. ‎
    </p>
</td>
                                            <td class="assignee">            coulibaly ELISE
    </td>
                                            <td class="reporter">            Diallo Nassirou Amadou Oumar
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 10:32 PM </td>
                                            <td class="updated"> 14/Mar/26 10:32 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>


                <tr id="issuerow10114" rel="10114" data-issuekey="SUP-39" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-39" href="https://grp31.atlassian.net/browse/SUP-39">SUP-39</a>
</td>
                                            <td class="summary"><p>
                TeamLead-Créer un Dockerfile pour conteneuriser l’application.
    </p>
</td>
                                            <td class="assignee">            coulibaly ELISE
    </td>
                                            <td class="reporter">            Diallo Nassirou Amadou Oumar
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 10:32 PM </td>
                                            <td class="updated"> 14/Mar/26 10:32 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>


                <tr id="issuerow10113" rel="10113" data-issuekey="SUP-38" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-38" href="https://grp31.atlassian.net/browse/SUP-38">SUP-38</a>
</td>
                                            <td class="summary"><p>
                TeamLead- Finaliser le README avec toutes les réponses aux questions critiques.
    </p>
</td>
                                            <td class="assignee">            coulibaly ELISE
    </td>
                                            <td class="reporter">            Diallo Nassirou Amadou Oumar
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 10:31 PM </td>
                                            <td class="updated"> 14/Mar/26 10:31 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>


                <tr id="issuerow10112" rel="10112" data-issuekey="SUP-37" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-37" href="https://grp31.atlassian.net/browse/SUP-37">SUP-37</a>
</td>
                                            <td class="summary"><p>
                TeamLead- Documenter l’ingénierie des prompts pour une tâche spécifique.
    </p>
</td>
                                            <td class="assignee">            coulibaly ELISE
    </td>
                                            <td class="reporter">            Diallo Nassirou Amadou Oumar
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 10:31 PM </td>
                                            <td class="updated"> 14/Mar/26 10:35 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>


                <tr id="issuerow10111" rel="10111" data-issuekey="SUP-36" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-36" href="https://grp31.atlassian.net/browse/SUP-36">SUP-36</a>
</td>
                                            <td class="summary"><p>
                TeamLead-Coordonner l’intégration des différentes branches.
    </p>
</td>
                                            <td class="assignee">            coulibaly ELISE
    </td>
                                            <td class="reporter">            Diallo Nassirou Amadou Oumar
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 10:30 PM </td>
                                            <td class="updated"> 14/Mar/26 10:30 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>


                <tr id="issuerow10110" rel="10110" data-issuekey="SUP-35" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-35" href="https://grp31.atlassian.net/browse/SUP-35">SUP-35</a>
</td>
                                            <td class="summary"><p>
                TeamLead- Mettre en place GitHub Actions (.github/workflows/ci.yml) avec un test minimal.
    </p>
</td>
                                            <td class="assignee">            coulibaly ELISE
    </td>
                                            <td class="reporter">            Diallo Nassirou Amadou Oumar
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 10:29 PM </td>
                                            <td class="updated"> 14/Mar/26 10:29 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>


                <tr id="issuerow10109" rel="10109" data-issuekey="SUP-34" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-34" href="https://grp31.atlassian.net/browse/SUP-34">SUP-34</a>
</td>
                                            <td class="summary"><p>
                TeamLead-Initialiser le README.md avec la description du projet.
    </p>
</td>
                                            <td class="assignee">            coulibaly ELISE
    </td>
                                            <td class="reporter">            Diallo Nassirou Amadou Oumar
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 10:28 PM </td>
                                            <td class="updated"> 14/Mar/26 10:28 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>


                <tr id="issuerow10108" rel="10108" data-issuekey="SUP-33" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-33" href="https://grp31.atlassian.net/browse/SUP-33">SUP-33</a>
</td>
                                            <td class="summary"><p>
                TeamLead-Configurer le tableau jira et inviter l’équipe
    </p>
</td>
                                            <td class="assignee">            coulibaly ELISE
    </td>
                                            <td class="reporter">            Diallo Nassirou Amadou Oumar
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 10:27 PM </td>
                                            <td class="updated"> 14/Mar/26 10:27 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>


                <tr id="issuerow10107" rel="10107" data-issuekey="SUP-32" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-32" href="https://grp31.atlassian.net/browse/SUP-32">SUP-32</a>
</td>
                                            <td class="summary"><p>
                TeamLead-Configurer le tableau jira et inviter l’équipe
    </p>
</td>
                                            <td class="assignee">            coulibaly ELISE
    </td>
                                            <td class="reporter">            Diallo Nassirou Amadou Oumar
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 10:26 PM </td>
                                            <td class="updated"> 14/Mar/26 10:26 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>


                <tr id="issuerow10106" rel="10106" data-issuekey="SUP-31" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-31" href="https://grp31.atlassian.net/browse/SUP-31">SUP-31</a>
</td>
                                            <td class="summary"><p>
                TeamLead-Créer le dépôt GitHub et la structure de dossiers
    </p>
</td>
                                            <td class="assignee">            coulibaly ELISE
    </td>
                                            <td class="reporter">            Diallo Nassirou Amadou Oumar
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 10:25 PM </td>
                                            <td class="updated"> 14/Mar/26 10:25 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>
        </tbody>
    
</table>

</details>

### 👤 Rôle : dataEngineer
<details>
<summary>Cliquez pour voir les tâches de dataEngineer</summary>

<table border="1">

                        <thead>
        <tr class="rowHeader">
            
                                                            <th class="colHeaderLink headerrow-issuetype" data-id="issuetype">
                                                        Issue Type
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-issuekey" data-id="issuekey">
                                                        Key
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-summary" data-id="summary">
                                                        Summary
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-assignee" data-id="assignee">
                                                        Assignee
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-reporter" data-id="reporter">
                                                        Reporter
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-priority" data-id="priority">
                                                        Priority
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-status" data-id="status">
                                                        Status
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-resolution" data-id="resolution">
                                                        Resolution
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-created" data-id="created">
                                                        Created
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-updated" data-id="updated">
                                                        Updated
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-duedate" data-id="duedate">
                                                        Due date
                                                    </th>
                                                                    </tr>
    </thead>
    <tbody>
                    

                <tr id="issuerow10098" rel="10098" data-issuekey="SUP-23" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-23" href="https://grp31.atlassian.net/browse/SUP-23">SUP-23</a>
</td>
                                            <td class="summary"><p>
                Data Engineering_Documenter les fonctions dans le code (docstrings).
    </p>
</td>
                                            <td class="assignee">            Diallo Nassirou Amadou Oumar
    </td>
                                            <td class="reporter">            Diallo Nassirou Amadou Oumar
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 10:12 PM </td>
                                            <td class="updated"> 14/Mar/26 10:12 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>


                <tr id="issuerow10097" rel="10097" data-issuekey="SUP-22" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-22" href="https://grp31.atlassian.net/browse/SUP-22">SUP-22</a>
</td>
                                            <td class="summary"><p>
                Data Engineering_ Écrire les tests unitaires dans tests/test_data_processing.py.
    </p>
</td>
                                            <td class="assignee">            Diallo Nassirou Amadou Oumar
    </td>
                                            <td class="reporter">            Diallo Nassirou Amadou Oumar
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 10:10 PM </td>
                                            <td class="updated"> 14/Mar/26 10:10 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>


                <tr id="issuerow10096" rel="10096" data-issuekey="SUP-21" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-21" href="https://grp31.atlassian.net/browse/SUP-21">SUP-21</a>
</td>
                                            <td class="summary"><p>
                Data Engineering_ Participer à la rédaction des sections README concernant les données
    </p>
</td>
                                            <td class="assignee">            Diallo Nassirou Amadou Oumar
    </td>
                                            <td class="reporter">            Diallo Nassirou Amadou Oumar
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 10:05 PM </td>
                                            <td class="updated"> 14/Mar/26 10:05 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>


                <tr id="issuerow10095" rel="10095" data-issuekey="SUP-20" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-20" href="https://grp31.atlassian.net/browse/SUP-20">SUP-20</a>
</td>
                                            <td class="summary"><p>
                Data Engineering - Creer pipeline de pretraitement complet (NA, encodage, normalisation)
    </p>
</td>
                                            <td class="assignee">            Diallo Nassirou Amadou Oumar
    </td>
                                            <td class="reporter">            coulibaly ELISE
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 6:30 PM </td>
                                            <td class="updated"> 14/Mar/26 6:30 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>


                <tr id="issuerow10094" rel="10094" data-issuekey="SUP-19" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-19" href="https://grp31.atlassian.net/browse/SUP-19">SUP-19</a>
</td>
                                            <td class="summary"><p>
                Data Engineering - Implementer optimize_memory(df) dans src/data_processing.py
    </p>
</td>
                                            <td class="assignee">            Diallo Nassirou Amadou Oumar
    </td>
                                            <td class="reporter">            coulibaly ELISE
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 6:22 PM </td>
                                            <td class="updated"> 14/Mar/26 6:22 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>


                <tr id="issuerow10093" rel="10093" data-issuekey="SUP-18" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-18" href="https://grp31.atlassian.net/browse/SUP-18">SUP-18</a>
</td>
                                            <td class="summary"><p>
                Data Engineering - Fournir un resume clair des conclusions a l equipe
    </p>
</td>
                                            <td class="assignee">            Diallo Nassirou Amadou Oumar
    </td>
                                            <td class="reporter">            coulibaly ELISE
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 6:20 PM </td>
                                            <td class="updated"> 14/Mar/26 6:20 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>


                <tr id="issuerow10092" rel="10092" data-issuekey="SUP-17" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-17" href="https://grp31.atlassian.net/browse/SUP-17">SUP-17</a>
</td>
                                            <td class="summary"><p>
                Data Engineering - Calculer matrice de correlation et identifier features importantes
    </p>
</td>
                                            <td class="assignee">            Diallo Nassirou Amadou Oumar
    </td>
                                            <td class="reporter">            coulibaly ELISE
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 6:18 PM </td>
                                            <td class="updated"> 14/Mar/26 6:18 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>
        </tbody>
    
</table>

</details>

### 👤 Rôle : dataAnalyst
<details>
<summary>Cliquez pour voir les tâches de dataAnalyst</summary>

<table border="1">

                        <thead>
        <tr class="rowHeader">
            
                                                            <th class="colHeaderLink headerrow-issuetype" data-id="issuetype">
                                                        Issue Type
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-issuekey" data-id="issuekey">
                                                        Key
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-summary" data-id="summary">
                                                        Summary
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-assignee" data-id="assignee">
                                                        Assignee
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-reporter" data-id="reporter">
                                                        Reporter
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-priority" data-id="priority">
                                                        Priority
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-status" data-id="status">
                                                        Status
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-resolution" data-id="resolution">
                                                        Resolution
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-created" data-id="created">
                                                        Created
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-updated" data-id="updated">
                                                        Updated
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-duedate" data-id="duedate">
                                                        Due date
                                                    </th>
                                                                    </tr>
    </thead>
    <tbody>
                    

                <tr id="issuerow10085" rel="10085" data-issuekey="SUP-10" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-10" href="https://grp31.atlassian.net/browse/SUP-10">SUP-10</a>
</td>
                                            <td class="summary"><p>
                EDA - Verification de l&#39;equilibre des classes
    </p>
</td>
                                            <td class="assignee">            Ange Sarah
    </td>
                                            <td class="reporter">            coulibaly ELISE
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 6:07 PM </td>
                                            <td class="updated"> 14/Mar/26 6:07 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>


                <tr id="issuerow10084" rel="10084" data-issuekey="SUP-9" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-9" href="https://grp31.atlassian.net/browse/SUP-9">SUP-9</a>
</td>
                                            <td class="summary"><p>
                EDA - Détection et traitement des outliers
    </p>
</td>
                                            <td class="assignee">            Ange Sarah
    </td>
                                            <td class="reporter">            coulibaly ELISE
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 6:07 PM </td>
                                            <td class="updated"> 14/Mar/26 6:07 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>


                <tr id="issuerow10083" rel="10083" data-issuekey="SUP-8" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-8" href="https://grp31.atlassian.net/browse/SUP-8">SUP-8</a>
</td>
                                            <td class="summary"><p>
                EDA - Analyse des valeurs manquantes
    </p>
</td>
                                            <td class="assignee">            Ange Sarah
    </td>
                                            <td class="reporter">            coulibaly ELISE
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 6:07 PM </td>
                                            <td class="updated"> 14/Mar/26 6:07 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>
        </tbody>
    
</table>

</details>

### 👤 Rôle : IA Engineer
<details>
<summary>Cliquez pour voir les tâches de IA Engineer</summary>

<table border="1">

                        <thead>
        <tr class="rowHeader">
            
                                                            <th class="colHeaderLink headerrow-issuetype" data-id="issuetype">
                                                        Issue Type
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-issuekey" data-id="issuekey">
                                                        Key
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-summary" data-id="summary">
                                                        Summary
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-assignee" data-id="assignee">
                                                        Assignee
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-reporter" data-id="reporter">
                                                        Reporter
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-priority" data-id="priority">
                                                        Priority
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-status" data-id="status">
                                                        Status
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-resolution" data-id="resolution">
                                                        Resolution
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-created" data-id="created">
                                                        Created
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-updated" data-id="updated">
                                                        Updated
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-duedate" data-id="duedate">
                                                        Due date
                                                    </th>
                                                                    </tr>
    </thead>
    <tbody>
                    

                <tr id="issuerow10105" rel="10105" data-issuekey="SUP-30" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-30" href="https://grp31.atlassian.net/browse/SUP-30">SUP-30</a>
</td>
                                            <td class="summary"><p>
                ML-Engineer-Fournir à AD les informations nécessaires pour l’intégration.
    </p>
</td>
                                            <td class="assignee">            MOHAMED JOUAHAR
    </td>
                                            <td class="reporter">            Diallo Nassirou Amadou Oumar
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 10:20 PM </td>
                                            <td class="updated"> 14/Mar/26 10:20 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>


                <tr id="issuerow10104" rel="10104" data-issuekey="SUP-29" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-29" href="https://grp31.atlassian.net/browse/SUP-29">SUP-29</a>
</td>
                                            <td class="summary"><p>
                ML-Engineer- Écrire les tests dans tests/test_model.py.
    </p>
</td>
                                            <td class="assignee">            MOHAMED JOUAHAR
    </td>
                                            <td class="reporter">            Diallo Nassirou Amadou Oumar
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 10:19 PM </td>
                                            <td class="updated"> 14/Mar/26 10:19 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>


                <tr id="issuerow10103" rel="10103" data-issuekey="SUP-28" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-28" href="https://grp31.atlassian.net/browse/SUP-28">SUP-28</a>
</td>
                                            <td class="summary"><p>
                ML-Engineer- Intégrer SHAP (valeurs, graphiques : summary plot, dependance plot, force plot).
    </p>
</td>
                                            <td class="assignee">            MOHAMED JOUAHAR
    </td>
                                            <td class="reporter">            Diallo Nassirou Amadou Oumar
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 10:18 PM </td>
                                            <td class="updated"> 14/Mar/26 10:18 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>


                <tr id="issuerow10102" rel="10102" data-issuekey="SUP-27" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-27" href="https://grp31.atlassian.net/browse/SUP-27">SUP-27</a>
</td>
                                            <td class="summary"><p>
                ML-Engineer- Sauvegarder le modèle final (apk) et le préprocesseur associé..
    </p>
</td>
                                            <td class="assignee">            MOHAMED JOUAHAR
    </td>
                                            <td class="reporter">            Diallo Nassirou Amadou Oumar
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 10:17 PM </td>
                                            <td class="updated"> 14/Mar/26 10:17 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>


                <tr id="issuerow10101" rel="10101" data-issuekey="SUP-26" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-26" href="https://grp31.atlassian.net/browse/SUP-26">SUP-26</a>
</td>
                                            <td class="summary"><p>
                ML-Engineer-Comparer les performances et sélectionner le meilleur modèle.
    </p>
</td>
                                            <td class="assignee">            MOHAMED JOUAHAR
    </td>
                                            <td class="reporter">            Diallo Nassirou Amadou Oumar
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 10:16 PM </td>
                                            <td class="updated"> 14/Mar/26 10:16 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>


                <tr id="issuerow10100" rel="10100" data-issuekey="SUP-25" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-25" href="https://grp31.atlassian.net/browse/SUP-25">SUP-25</a>
</td>
                                            <td class="summary"><p>
                ML-Engineer- Implémenter l’entraînement et l’évaluation avec validation croisée.
    </p>
</td>
                                            <td class="assignee">            MOHAMED JOUAHAR
    </td>
                                            <td class="reporter">            Diallo Nassirou Amadou Oumar
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 10:15 PM </td>
                                            <td class="updated"> 14/Mar/26 10:15 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>


                <tr id="issuerow10099" rel="10099" data-issuekey="SUP-24" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-24" href="https://grp31.atlassian.net/browse/SUP-24">SUP-24</a>
</td>
                                            <td class="summary"><p>
                ML-Engineer- Entraîner au moins trois modèles (SVM, Random Forest, LightGBM, CatBoost
    </p>
</td>
                                            <td class="assignee">            MOHAMED JOUAHAR
    </td>
                                            <td class="reporter">            Diallo Nassirou Amadou Oumar
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 10:14 PM </td>
                                            <td class="updated"> 14/Mar/26 10:14 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>
        </tbody>
    
</table>

</details>

### 👤 Rôle : Web dev
<details>
<summary>Cliquez pour voir les tâches de Web dev</summary>

<table border="1">

                        <thead>
        <tr class="rowHeader">
            
                                                            <th class="colHeaderLink headerrow-issuetype" data-id="issuetype">
                                                        Issue Type
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-issuekey" data-id="issuekey">
                                                        Key
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-summary" data-id="summary">
                                                        Summary
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-assignee" data-id="assignee">
                                                        Assignee
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-reporter" data-id="reporter">
                                                        Reporter
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-priority" data-id="priority">
                                                        Priority
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-status" data-id="status">
                                                        Status
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-resolution" data-id="resolution">
                                                        Resolution
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-created" data-id="created">
                                                        Created
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-updated" data-id="updated">
                                                        Updated
                                                    </th>
                                                
                                                            <th class="colHeaderLink headerrow-duedate" data-id="duedate">
                                                        Due date
                                                    </th>
                                                                    </tr>
    </thead>
    <tbody>
                    

                <tr id="issuerow10091" rel="10091" data-issuekey="SUP-16" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-16" href="https://grp31.atlassian.net/browse/SUP-16">SUP-16</a>
</td>
                                            <td class="summary"><p>
                Streamlit App - Tester l’application manuellement
    </p>
</td>
                                            <td class="assignee">            Sanogo
    </td>
                                            <td class="reporter">            coulibaly ELISE
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 6:17 PM </td>
                                            <td class="updated"> 14/Mar/26 6:17 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>


                <tr id="issuerow10090" rel="10090" data-issuekey="SUP-15" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-15" href="https://grp31.atlassian.net/browse/SUP-15">SUP-15</a>
</td>
                                            <td class="summary"><p>
                Streamlit App - Intégrer visualisations SHAP
    </p>
</td>
                                            <td class="assignee">            Sanogo
    </td>
                                            <td class="reporter">            coulibaly ELISE
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 6:17 PM </td>
                                            <td class="updated"> 14/Mar/26 6:17 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>


                <tr id="issuerow10089" rel="10089" data-issuekey="SUP-14" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-14" href="https://grp31.atlassian.net/browse/SUP-14">SUP-14</a>
</td>
                                            <td class="summary"><p>
                Streamlit App - Afficher probabilité et classe prédite
    </p>
</td>
                                            <td class="assignee">            Sanogo
    </td>
                                            <td class="reporter">            coulibaly ELISE
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 6:17 PM </td>
                                            <td class="updated"> 14/Mar/26 6:17 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>


                <tr id="issuerow10088" rel="10088" data-issuekey="SUP-13" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-13" href="https://grp31.atlassian.net/browse/SUP-13">SUP-13</a>
</td>
                                            <td class="summary"><p>
                Streamlit App - Charger modèle et préprocesseur
    </p>
</td>
                                            <td class="assignee">            Sanogo
    </td>
                                            <td class="reporter">            coulibaly ELISE
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 6:11 PM </td>
                                            <td class="updated"> 14/Mar/26 6:11 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>


                <tr id="issuerow10087" rel="10087" data-issuekey="SUP-12" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-12" href="https://grp31.atlassian.net/browse/SUP-12">SUP-12</a>
</td>
                                            <td class="summary"><p>
                Streamlit App - Concevoir interface utilisateur
    </p>
</td>
                                            <td class="assignee">            Sanogo
    </td>
                                            <td class="reporter">            coulibaly ELISE
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 6:11 PM </td>
                                            <td class="updated"> 14/Mar/26 6:11 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>


                <tr id="issuerow10086" rel="10086" data-issuekey="SUP-11" class="issuerow">
                                            <td class="issuetype">    Task
</td>
                                            <td class="issuekey">

    <a class="issue-link" data-issue-key="SUP-11" href="https://grp31.atlassian.net/browse/SUP-11">SUP-11</a>
</td>
                                            <td class="summary"><p>
                Streamlit App - Develop main application (app/app.py)
    </p>
</td>
                                            <td class="assignee">            Sanogo
    </td>
                                            <td class="reporter">            coulibaly ELISE
    </td>
                                            <td class="priority">           Medium
    </td>
                                            <td class="status">
                <span class=" jira-issue-status-lozenge aui-lozenge jira-issue-status-lozenge-blue-gray jira-issue-status-lozenge-new aui-lozenge-subtle jira-issue-status-lozenge-max-width-medium" data-tooltip="&lt;span class=&quot;jira-issue-status-tooltip-title&quot;&gt;Open&lt;/span&gt;&lt;br&gt;&lt;span class=&quot;jira-issue-status-tooltip-desc&quot;&gt;The issue is open and ready for the assignee to start work on it.&lt;/span&gt;">Open</span>    </td>
                                            <td class="resolution">    <em>Unresolved</em>
</td>
                                            <td class="created"> 14/Mar/26 6:09 PM </td>
                                            <td class="updated"> 14/Mar/26 6:09 PM </td>
                                            <td class="duedate">    &nbsp;
</td>
                    </tr>
        </tbody>
    
</table>

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
