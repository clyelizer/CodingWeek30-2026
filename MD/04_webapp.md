# 04 — Interface Web (FastAPI + SHAP)

## Contexte

L'application web `app/app.py` a été enrichie pour offrir une expérience utilisateur complète : authentification, page d'accueil, et console de diagnostic.

---

## Architecture technique

## Architecture technique

```
Streamlit (app/app.py)

Bibliothèques :
  streamlit          → interface utilisateur et serveur web
  joblib             → chargement du modèle (.pkl)
  shap               → explicabilité du modèle
  matplotlib         → affichage des graphiques SHAP
```

---

## Chargement des ressources

Au démarrage, l'application charge :
1. **Le modèle et le préprocesseur (.pkl)**.
2. **Les colonnes features** attendues par le pipeline.

---

## Flux de traitement

1. **Saisie des 10 paramètres** via le formulaire interactif de la barre latérale.
2. **Normalisation des données** via `preprocessor.pkl`.
3. **Prédiction** lancée au clic sur le bouton "Lancer l'analyse".
4. **Affichage du résultat** du risque (Très faible à Élevé).
5. **Génération des Explications SHAP** et affichage du graphique waterfall approprié à la classe prédite.

---

## Migration depuis FastAPI (Archivé)

L'ancienne architecture basée sur FastAPI avec des vues HTML (`templates/`) et des assets (`static/`) a été **archivée** dans le dossier `archive/`. 
Cette refonte vers Streamlit permet une intégration plus native, rapide et interactive avec les modèles de Machine Learning (notamment pour l'affichage des graphiques dynamiques générés par Matplotlib/SHAP).

