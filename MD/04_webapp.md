# 04 — Interface Web (FastAPI + SHAP)

## Contexte

L'application web `app/app.py` constitue l'interface utilisateur finale du
système. Elle permet à un praticien de saisir les 10 paramètres cliniques d'un
enfant et d'obtenir instantanément :
- La **probabilité d'appendicite** calculée par le Random Forest
- La **décision** recommandée (appendicite probable / peu probable)
- L'**explication SHAP** sous forme de graphique waterfall

---

## Architecture technique

```
FastAPI
  ├── GET  /         → landing page (landing_page.html)
  ├── GET  /login    → authentification (auth.html)
  ├── GET  /form     → console diagnostic (diagnosis_console.html)
  └── POST /predict  → console diagnostic mise à jour (diagnosis_console.html)

Bibliothèques :
  fastapi + uvicorn  → serveur ASGI
  Jinja2             → templates HTML
  python-multipart   → parsing des données de formulaire
  joblib             → chargement du modèle RF
```

**Lancement :**
```bash
# Depuis la racine du projet
uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload

# Ou directement
python app/app.py --port 8000 --reload
```

L'application démarre sur `http://localhost:8000`.

---

## Chargement des ressources au démarrage

Au démarrage du serveur, deux ressources sont chargées **une seule fois** (singleton) :

1. **Le modèle Random Forest** : `models/random_forest.joblib`
   - Pipeline sklearn complet (StandardScaler + RandomForestClassifier)
   - AUC-ROC = 0.9287 sur le jeu de test

2. **Les données de référence** : `data/processed/processed_data.joblib`
   - `feature_cols` : liste des 10 colonnes dans l'ordre attendu par le modèle
   - Médianes du jeu de test → valeurs par défaut du formulaire

**Médianes utilisées comme valeurs par défaut :**

| Feature | Médiane test set |
|---------|-----------------|
| `Lower_Right_Abd_Pain` | 1.0 (Oui) |
| `Migratory_Pain` | 0.0 (Non) |
| `Body_Temperature` | 37.2 °C |
| `WBC_Count` | 11.25 G/L |
| `CRP` | 7.0 mg/L |
| `Neutrophil_Percentage` | 75.5 % |
| `Ipsilateral_Rebound_Tenderness` | 0.0 (Non) |
| `Appendix_Diameter` | 7.5 mm |
| `Nausea` | 1.0 (Oui) |
| `Age` | 11.23 ans |

---

## Flux de traitement d'une prédiction

```
POST /predict
  │
  ├─ 1. Réception des 10 champs du formulaire (FastAPI Form(...))
  │
  ├─ 2. Construction du vecteur features (_build_input_row)
  │       - Champs numériques  : cast float, fallback médiane si invalide
  │       - Champs binaires    : "yes" → 1.0, "no" → 0.0
  │       → DataFrame 1 ligne × 10 colonnes (ordre = feature_cols)
  │
  ├─ 3. Prédiction (predict_proba_safe)
  │       → probabilité ∈ [0, 1]
  │
  ├─ 4. Explication SHAP (compute_shap_values + make_shap_waterfall_b64)
  │       → image PNG encodée base64 (non-fatale si erreur)
  │
  └─ 5. Rendu diagnosis_console.html
          - prob      : probabilité en %
          - decision  : "appendicite" / "pas d'appendicite"
          - risk_class: "danger" / "success" (couleur Bootstrap)
          - shap_b64  : image inline base64
```

---

## Formulaire de saisie (`app/templates/diagnosis_console.html`)

Le formulaire est organisé en 4 sections cliniques :

### §1 — Patient
| Champ HTML | Feature modèle | Unité |
|-----------|---------------|-------|
| `age` | `Age` | ans |
| `body_temperature` | `Body_Temperature` | °C |

### §2 — Examen clinique
| Champ HTML | Feature modèle | Encodage |
|-----------|---------------|----------|
| `lower_right_abd_pain` | `Lower_Right_Abd_Pain` | yes→1 / no→0 |
| `migratory_pain` | `Migratory_Pain` | yes→1 / no→0 |
| `nausea` | `Nausea` | yes→1 / no→0 |
| `ipsilateral_rebound_tenderness` | `Ipsilateral_Rebound_Tenderness` | yes→1 / no→0 |

### §3 — Biologie
| Champ HTML | Feature modèle | Unité |
|-----------|---------------|-------|
| `wbc_count` | `WBC_Count` | G/L |
| `neutrophil_percentage` | `Neutrophil_Percentage` | % |
| `crp` | `CRP` | mg/L |

### §4 — Échographie
| Champ HTML | Feature modèle | Unité |
|-----------|---------------|-------|
| `appendix_diameter` | `Appendix_Diameter` | mm |

---

## Affichage du résultat (`app/templates/diagnosis_console.html`)

La page résultat affiche :

**Probabilité d'appendicite (grand affichage)**
- Chiffre en % avec code couleur Bootstrap :
  - `text-danger` (rouge) si ≥ 50% → appendicite probable
  - `text-success` (vert) si < 50% → appendicite peu probable

**Message clinique contextualisé**
- ≥ 50% : *"⚠️ Appendicite probable — prise en charge chirurgicale urgente à envisager"*
- < 50% : *"✅ Appendicite peu probable — surveillance clinique suffisante"*

**Graphique SHAP waterfall**  
Visualise les contributions positives (rouges) et négatives (bleues) de chaque
variable à la prédiction. Permet au praticien de comprendre **pourquoi** le
modèle a prédit ce score.

**Rappel modèle :** `Random Forest (AUC = 0.9287 sur le jeu de test, n=776)`

---

## Décisions de conception

**Design unifie applique sur les pages hors landing :**
Le projet utilisait des styles disperses selon les templates (`auth`, `diagnosis_console`, pages legacy), ce qui degradait la lisibilite et la maintenance. Le choix retenu est un **theme centralise** dans `app/static/css/unified_theme.css` (tokens couleur, typographie, cartes, boutons, formulaires), afin d'obtenir :
- une experience visuelle coherente entre les ecrans,
- un cout de maintenance reduit (modification en un seul point),
- une meilleure evolutivite (ajout de pages sans dupliquer du CSS).

**Header partage (partial Jinja) :**
Le header est factorise dans `app/templates/partials/shared_header.html` et inclus dans les pages applicatives (`auth.html`, `diagnosis_console.html`).
- mode public : boutons Accueil/Connexion,
- mode connecte : actions metier (nouvelle consultation, historique, PDF, deconnexion),
- un seul composant a maintenir pour la navigation applicative.

**Gestion de l'authentification :**
Auth basee sur session signee, sans JWT externe :
- verification login/mot de passe contre SQLite (`users`),
- hash mot de passe via PBKDF2 (`sha256`, sel aleatoire, 100k iterations),
- cookie `pedi_session` signe HMAC et expire (4h),
- controles d'acces sur `/form`, `/predict`, `/api/predict`, `/api/history`.

**Identifiants de test (local) :**
Un compte admin est cree automatiquement a l'initialisation DB si absent :
- username : `admin`
- password : `admin123`

Ce compte est destine aux tests locaux et doit etre remplace en environnement de production.

**FastAPI plutôt que Flask :**  
FastAPI offre la validation automatique des types via Pydantic, supporte async
nativement, et génère automatiquement une documentation OpenAPI accessible
sur `/docs`.

**Singleton modèle au démarrage :**  
Charger le modèle à chaque requête POST coûterait ~200ms inutiles.
Le chargement unique au démarrage garantit des temps de réponse < 50ms hors SHAP.

**SHAP non-fatal :**  
Le calcul SHAP peut prendre 1–2s et peut échouer sur certaines configurations.
L'application retourne toujours un résultat même sans le graphique SHAP.

**Encodage oui/non → 1/0 en amont :**  
L'encodage est réalisé dans `_build_input_row()` dans `app.py` et non dans le
template HTML (qui envoie "yes"/"no"). Cela préserve la lisibilité du formulaire
et centralise la logique de transformation dans le code Python.

---

## Statut des tâches

| Composant | Fichier | Statut |
|-----------|---------|--------|
| Backend FastAPI | `app/app.py` | ✅ Complet |
| Formulaire de saisie | `app/templates/diagnosis_console.html` | ✅ Réécrit (10 features) |
| Page résultat | `app/templates/diagnosis_console.html` | ✅ AUC corrigé (0.9287) |
| Démarrage uvicorn | `uvicorn app.app:app --port 8000` | ✅ Fonctionne |
