# 04 — Interface Web (FastAPI + SHAP)

## Contexte

L'application web `app/app.py` a été enrichie pour offrir une expérience utilisateur complète : authentification, page d'accueil, et console de diagnostic.

---

## Architecture technique

```
FastAPI
  ├── GET  /                 → Page d'accueil (landing_page.html)
  ├── GET  /login            → Authentification (auth.html)
  ├── GET  /form             → Console de diagnostic (diagnosis_form.html)
  ├── POST /api/predict      → Résultat (diagnosis_result.html)

Bibliothèques :
  fastapi + uvicorn  → serveur ASGI
  Jinja2             → templates HTML
  joblib             → chargement du modèle (.pkl)
```

---

## Chargement des ressources

Au démarrage, l'application charge :
1. **Le modèle et le préprocesseur (.pkl)**.
2. **Les colonnes features** attendues par le pipeline.

---

## Flux de traitement

1. **Authentification** (optionnelle selon session).
2. **Saisie des 10 paramètres** dans la console de diagnostic.
3. **Prédiction et Explications SHAP** générées en temps réel.
4. **Affichage du résultat** avec graphique waterfall.

---

## Templates HTML (`app/templates/`)

La structure des templates a été modernisée :
- `landing_page.html` : Présentation du projet.
- `auth.html` : Connexion/Inscription avec animation flip.
- `diagnosis_form.html` : Saisie des 10 features cliniques.
- `diagnosis_result.html` : Résultat + Graphique SHAP.
- `partials/` : En-tête partagé pour une navigation fluide.

---

## Décisions de conception

**Authentification par session :** Utilisation de cookies signés (HMAC) pour une gestion simple et sécurisée des sessions.

**Design unifié :** Utilisation d'un thème CSS global (`unified_theme.css`) pour une esthétique premium et cohérente.
