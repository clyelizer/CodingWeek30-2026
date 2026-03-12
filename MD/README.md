# MD — Journal des Rapports Techniques

**Projet :** PediAppendix — Diagnostic pédiatrique de l'appendicite (ML + SHAP)  
**Dataset :** UCI Regensburg Pediatric Appendicitis  
**Stack :** Python 3.14 · FastAPI · Random Forest · SHAP · pytest

---

## Index des rapports

| # | Fichier | Sujet | Date | Commits |
|:---:|---|---|---|---|
| 01 | [01_pipeline_quality_corrections.md](01_pipeline_quality_corrections.md) | Audit qualité pipeline + corrections leakage + fix encodage | 12/03/2026 | `e9b1a84` `7e60157` |
| 02 | [02_correlation_audit_form_expansion.md](02_correlation_audit_form_expansion.md) | Étude corrélation + audit forme + expansion 16→35 inputs | 12/03/2026 | `d063f9e` |

---

## État du projet (12/03/2026)

### Modèles entraînés

| Modèle | AUC | F1 | Accuracy |
|---|:---:|:---:|:---:|
| **Random Forest** (défaut app) | **0.9724** | 0.9319 | 0.9167 |
| LightGBM | 0.9623 | **0.9424** | **0.9295** |
| CatBoost | 0.9717 | 0.9263 | 0.9103 |
| SVM | 0.9616 | 0.9032 | 0.8846 |

### Dataset
- Source : `data/external/$RTQ57LQ.xlsx`
- Processé : `data/processed/app_data_final.xlsx` (776×59)
- Split : 620 train / 156 test (80/20, stratifié)

### Tests
- **67/67 passent** (42 data_processing + 25 model)

### Infrastructure
- FastAPI sur port 8000
- CI/CD GitHub Actions (test + lint + docker)
- Dockerfile multi-stage

---

## Convention de nommage des rapports

```
NN_sujet_court.md
```
- `NN` : numéro séquentiel à deux chiffres
- Sujet en snake_case anglais ou français
- Un rapport par session de travail ou par thème majeur
