# PediAppendix — Documentation synthétique

Ce README reprend et condense la documentation technique présente dans le
répertoire `MD/`. Il décrit l'architecture, le pipeline de données, les choix
méthodologiques et les commandes essentielles pour exécuter le projet.

Résumé : modèle de production = **Random Forest** (AUC-ROC = **0.9287** sur
le jeu de test). Dataset : Regensburg Pediatric Appendicitis (UCI), n≈776.

---

## Index rapide 

- Documentation détaillée : [MD/README.md](MD/README.md)
- Pipeline de preprocessing : [MD/01_data_processing.md](MD/01_data_processing.md)
- Entraînement & sélection : [MD/02_train_model.md](MD/02_train_model.md)
- Évaluation & SHAP : [MD/03_evaluate_model.md](MD/03_evaluate_model.md)
- Interface web : [MD/04_webapp.md](MD/04_webapp.md)

---

## Arborescence clé

```
.
├── app/                   # FastAPI + templates
├── data/                  # raw/ et processed/
├── models/                # .joblib (modèles et preprocessors)
├── src/                   # data_processing.py, train_model.py, evaluate_model.py
├── tests/                 # suite pytest (34 tests)
└── MD/                    # documentation technique détaillée
```

---

## Pipeline de données (résumé)

1. Chargement des données brutes depuis `data/raw/`.
2. Nettoyage et optimisation mémoire (`optimize_memory`).
3. Encodage / imputation : median pour numériques, mode pour catégoriques.
4. Split stratifié train/test (80/20).
5. Prétraitement via `ColumnTransformer` : `StandardScaler` pour numériques
   et `OneHotEncoder(handle_unknown='ignore')` pour catégoriques.
6. Sauvegarde des artefacts : `models/preprocessor.pkl` et
   `data/processed/processed_data.joblib`.

Décisions clés : StandardScaler dans chaque Pipeline pour éviter toute data
leakage ; OneHotEncoding pour variables nominales ; import lazy de `shap`
pour éviter le coût d'import au démarrage.

---

## Modèles et métriques

Modèles entraînés (exemples) : Logistic Regression, Random Forest,
Gradient Boosting, SVM (RBF). Chaque modèle est encapsulé dans un
`sklearn.Pipeline` contenant le scaler puis le classifieur.

Métrique principale : **AUC-ROC** (robuste au choix du seuil en contexte médical).
Métriques secondaires : F1 macro, accuracy.

Résultats synthétiques (jeu test n≈156) :

|                Modèle | AUC-ROC |
| ---------------------: | :-----: |
| Random Forest (retenu) | 0.9287 |
|      Gradient Boosting | 0.9141 |
|    Logistic Regression | 0.8283 |
|              SVM (RBF) | 0.8102 |

---

## Interface web (FastAPI)

- Points d'accès principaux :

  - `GET  /` → landing page
  - `GET  /login` → authentification
  - `GET  /form` → formulaire de saisie
  - `POST /predict` → calcul et rendu du résultat
- Chargement singleton au démarrage : modèle production
  (`models/random_forest.joblib`) et `data/processed/...` (feature_cols,
  médianes utilisées comme valeurs par défaut du formulaire).
- SHAP : calcul non-fatal — si la génération échoue, l'interface rend quand
  même la probabilité sans le graphique explicatif.

---

## Commandes essentielles

Installation :

```bash
pip install -r requirements.txt
```

Traitement des données :

```bash
python src/data_processing.py
# --> data/processed/processed_data.joblib
```

Entraînement :

```bash
python src/train_model.py
# --> models/*.joblib
```

Tests :

```bash
python -m pytest tests/ -q
# 34 passed
```

Lancer l'app (dev) :

```bash
uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
# puis ouvrir http://localhost:8000
```

---

## Points d'attention et recommandations

- Vérifier que la colonne cible `Diagnosis` est encodée numériquement avant
  d'exécuter le pipeline (`0/1`). Si elle est textuelle, utiliser
  `LabelEncoder()` ou mapping explicite.
- Ajouter `'Segmented_Neutrophils'` aux features numériques si présent dans
  vos données (forte corrélation observée dans l'EDA).
- Pour gérer plus finement le déséquilibre de classes : SMOTE ou réglage
  des poids, mais `class_weight='balanced'` a été appliqué sur plusieurs modèles.

---

## Avertissement

Ce projet est un outil de recherche / démonstration. Il ne remplace pas le
jugement clinique et ne doit pas être utilisé en production sans validation
clinique et conformité aux réglementations en vigueur.

---

Pour la documentation technique complète et les justifications détaillées,
consultez les fichiers du dossier [`MD/`](MD/README.md).
