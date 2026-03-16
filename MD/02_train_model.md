# 02 — Entraînement et sélection des modèles

## Contexte

L'objectif est d'entraîner et de sélectionner les meilleurs modèles ML.

**Jeu de données :**
- Entraînement : **620 patients**
- Test : **156 patients**

---

## Modèles et Préprocesseur

Les modèles sont sauvegardés au format `.pkl` dans le dossier `models/`. Un préprocesseur (`preprocessor.pkl`) est utilisé pour harmoniser les données avant la prédiction.

```
models/
  preprocessor.pkl           ← Préprocesseur (StandardScaler + Encodage)
  random_forest_model.pkl    ← Modèle final
  ...
```

---

## Métriques choisies

**AUC-ROC :** Mesure la capacité de discrimination du modèle.

---

## Résultats (Jeu de test)

Le modèle **Random Forest** obtient les meilleurs résultats avec une **AUC-ROC de ~0.92**.

---

## Tests unitaires (`tests/test_model.py`)

Les tests vérifient :
1. Le chargement correct du modèle et du préprocesseur.
2. La validité des prédictions sur des données factices.

```
pytest tests/test_model.py -v
2 passed
```
