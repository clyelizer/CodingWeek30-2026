# Rapport 02 — Étude de Corrélation & Audit du Formulaire App

**Projet :** PediAppendix  
**Date :** 12 mars 2026  
**Commit :** `d063f9e`

---

## 1. Contexte

Deux questions ont déclenché cet audit :
1. L'étude de corrélation avait-elle été faite ?
2. Les valeurs d'entrée du formulaire web étaient-elles choisies selon les bons patterns ?

---

## 2. Étude de Corrélation

### État initial du notebook `eda.ipynb`

La section **5. Corrélations** existait dans le notebook (cellule `#VSC-8a8ba618`) mais **n'avait jamais été exécutée** (kernel Python 3.14 local vs. kernel Colab attendu).

### Code d'analyse corrélation (existant dans cellule #VSC-8a8ba618)

```python
df2 = df_processed.copy()
df2['target'] = (df2['Diagnosis'].str.strip() == 'appendicitis').astype(int)

# Exclure colonnes leakage
exclude_leakage = [
    'Diagnosis', 'Diagnosis_Presumptive', 'Severity', 'Management',
    'Length_of_Stay', 'Alvarado_Score', 'Paedriatic_Appendicitis_Score',
    'Perforation', 'Appendicular_Abscess', 'Sex', 'Stool'
]
num_cols = [c for c in df2.select_dtypes(include=[np.number]).columns
            if c not in exclude_leakage and c != 'target']

corr_matrix = df2[num_cols + ['target']].corr()
corr_target = corr_matrix['target'].drop('target').sort_values(ascending=False)

# Vérification multicolinéarité (paires |r| > 0.75)
upper = corr_feat.where(np.triu(np.ones(corr_feat.shape), k=1).astype(bool))
high_corr = [(upper.columns[j], upper.index[i], upper.iloc[i, j])
             for i in range(len(upper.index))
             for j in range(len(upper.columns))
             if upper.iloc[i, j] > 0.75]
```

**Résultat :** Pas d'exécution possible — le kernel du notebook est configuré pour Colab, non pour Python 3.14 local. La corrélation n'a pas été calculée en session.

---

## 3. Audit de la Feature Importance RF

### Méthode

Script d'audit `_importance.py` (temporaire, supprimé après usage) :

```python
import joblib, numpy as np, pandas as pd
d = joblib.load('models/test_data.joblib')
X_test = d['X_test']

from src.train_model import load_model
rf = load_model('models/random_forest.joblib')

# RF MDI (Mean Decrease in Impurity) feature importances
importances = pd.Series(rf.feature_importances_, index=X_test.columns)
importances = importances.sort_values(ascending=False)
```

### Résultats — 48 features, couverture formulaire initiale

```
#   Feature                                  RF Importance   Status
--------------------------------------------------------------------
1   Appendix_Diameter                           0.1917       IN FORM
2   WBC_Count                                   0.0781       IN FORM
3   Appendix_on_US                              0.0769       IN FORM
4   CRP_log                                     0.0649       IN FORM
5   Neutrophil_Percentage                       0.0493       IN FORM
6   Peritonitis                                 0.0476       MISSING  ← #6!
7   Surrounding_Tissue_Reaction                 0.0402       MISSING  ← #7!
8   Age                                         0.0305       IN FORM
9   Weight                                      0.0290       MISSING
10  Thrombocyte_Count                           0.0288       MISSING
11  BMI                                         0.0283       IN FORM
12  Body_Temperature                            0.0280       IN FORM
13  Hemoglobin                                  0.0280       IN FORM
14  Appendix_Wall_Layers                        0.0278       MISSING
15  Height                                      0.0270       MISSING
16  RBC_Count                                   0.0252       MISSING
17  RDW                                         0.0232       MISSING
18  Neutrophilia                                0.0167       MISSING
19  Contralateral_Rebound_Tenderness            0.0094       MISSING
20  Ketones_in_Urine                            0.0090       MISSING
... [et 28 autres features]

Total in form: 16/48
Importance covered by form features: 60.83%
Importance NOT covered: 39.17%
```

### Diagnostic

Le formulaire ne collectait que **16/48 features** = couverture de **60.8%** de l'importance RF. Des features au rang #6 et #7 (Péritonite, Réaction péri-appendiculaire) étaient absentes.

---

## 4. Corrections Apportées

### 4.1 `app/app.py` — `_encode_form()` : ajout sparse binaries

**Avant :** Les sparse binaires n'étaient jamais modifiées par le formulaire (commentaire "déjà à 0")

```python
# --- Sparse binaires (0/1) — non modifiées par l'utilisateur → déjà à 0
# (la plupart de ces colonnes valent 0 par défaut dans le dataset)
```

**Après :**
```python
# --- Sparse binaires exposées dans le formulaire (écho + signes rares)
for field in ("Surrounding_Tissue_Reaction", "Appendix_Wall_Layers",
              "Appendicolith", "Target_Sign", "Pathological_Lymph_Nodes",
              "Bowel_Wall_Thickening", "Meteorism"):
    val = data.get(field.lower(), "")
    if val in _YES_NO:
        row[field] = _YES_NO[val]
```

### 4.2 `app/app.py` — POST endpoint étendu

**Avant (16 paramètres) :**
```python
async def predict(
    request: Request,
    age: float = Form(default=10.0),
    sex: str = Form(default="male"),
    bmi: str = Form(default=""),
    body_temperature: float = Form(default=37.5),
    wbc_count: float = Form(default=11.0),
    crp: float = Form(default=10.0),
    hemoglobin: float = Form(default=12.5),
    neutrophil_percentage: float = Form(default=65.0),
    appendix_diameter: str = Form(default=""),
    us_performed: str = Form(default="yes"),
    appendix_on_us: str = Form(default="no"),
    migratory_pain: str = Form(default="no"),
    lower_right_abd_pain: str = Form(default="yes"),
    ipsilateral_rebound_tenderness: str = Form(default="no"),
    nausea: str = Form(default="no"),
    loss_of_appetite: str = Form(default="no"),
) -> HTMLResponse:
```

**Après (35 paramètres, organisés par catégorie) :**
```python
async def predict(
    request: Request,
    # --- Démographie
    age: float = Form(default=10.0),
    sex: str = Form(default="male"),
    weight: str = Form(default=""),
    height: str = Form(default=""),
    bmi: str = Form(default=""),
    # --- Signes vitaux
    body_temperature: float = Form(default=37.5),
    # --- Biologie
    wbc_count: float = Form(default=11.0),
    crp: float = Form(default=10.0),
    hemoglobin: float = Form(default=12.5),
    neutrophil_percentage: float = Form(default=65.0),
    thrombocyte_count: str = Form(default=""),
    rbc_count: str = Form(default=""),
    rdw: str = Form(default=""),
    segmented_neutrophils: str = Form(default=""),
    # --- Examen clinique
    migratory_pain: str = Form(default="no"),
    lower_right_abd_pain: str = Form(default="yes"),
    ipsilateral_rebound_tenderness: str = Form(default="no"),
    contralateral_rebound_tenderness: str = Form(default="no"),
    coughing_pain: str = Form(default="no"),
    peritonitis: str = Form(default="no"),
    psoas_sign: str = Form(default="no"),
    nausea: str = Form(default="no"),
    loss_of_appetite: str = Form(default="no"),
    neutrophilia: str = Form(default="no"),
    stool: str = Form(default="normal"),
    # --- Échographie
    appendix_diameter: str = Form(default=""),
    us_performed: str = Form(default="yes"),
    appendix_on_us: str = Form(default="no"),
    free_fluids: str = Form(default="no"),
    surrounding_tissue_reaction: str = Form(default="no"),
    appendix_wall_layers: str = Form(default="no"),
    appendicolith: str = Form(default="no"),
    # --- Urines
    rbc_in_urine: str = Form(default="no"),
    ketones_in_urine: str = Form(default="no"),
    wbc_in_urine: str = Form(default="no"),
) -> HTMLResponse:
```

### 4.3 `app/templates/index.html` — 6 sections complètes

Structure finale du formulaire :

| Section | Champs ajoutés | Champs totaux |
|---|---|:---:|
| Démographie | Poids, Taille | 5 |
| Signes vitaux | — | 1 |
| Biologie | Plaquettes, GR/RBC, RDW, Neutrophiles segmentés | 8 |
| Examen clinique | Rebond contra., Toux, **Péritonite**, Psoas, Neutrophilie, Selles | 11 |
| Échographie | Épanchement, **Réaction péri-appendiculaire**, Paroi, Appendicolithe | 7 |
| Analyse urinaire | Hématies, Cétones, GB urines | 3 |
| **TOTAL** | **+19** | **35** |

---

## 5. Impact sur la couverture d'importance

| Métrique | Avant | Après |
|---|:---:|:---:|
| Features collectées | 16/48 | 35/48 |
| Couverture importance RF | 60.8% | ~95% |
| Rang #6 Peritonitis (4.76%) | ❌ | ✅ |
| Rang #7 Surrounding_Tissue_Reaction (4.02%) | ❌ | ✅ |
| Rang #9 Weight (2.90%) | ❌ | ✅ |
| Rang #10 Thrombocyte_Count (2.88%) | ❌ | ✅ |

---

## 6. Analyse corrélation non exécutée — Actions restantes

Le code Pearson existe dans `notebooks/eda.ipynb` cellule `#VSC-8a8ba618`. Pour l'exécuter :

```bash
# Option A : kernel local Python 3.14
jupyter notebook --notebook-dir="c:\Users\clyel\Desktop\CodingWeek\notebooks"

# Option B : Colab — uploader data/external/$RTQ57LQ.xlsx
```

La corrélation à calculer :
- Pearson par rapport à `Diagnosis` (binaire)
- Détection multicolinéarité (paires |r| > 0.75)
- Heatmap des 14 variables les plus corrélées

---

## 7. Commit

| Hash | Message |
|---|---|
| `d063f9e` | feat: expand form to 35 inputs (top features by RF importance) + fix sparse binary encoding |
