# PROMPTS.md

Ce document recense les invites (prompts) utilisées avec des AI-assistants (Copilot / ChatGPT) durant le projet. Il doit être enrichi au fil des itérations.

Exemple 1 — optimisation mémoire (`optimize_memory`)
- Prompt utilisé :

```
You are a Python expert. I need a function optimize_memory(df) that reduces pandas DataFrame memory by downcasting integer and float columns and converting low-cardinality object columns to category. Provide a function that prints memory before and after and returns the optimized DataFrame. Use numpy and pandas.
```

- Résultat attendu : fonction qui
  - downcaste int64→int8/int16/int32 quand possible,
  - convertit float64→float32 quand possible,
  - convertit object→category si cardinalité faible,
  - affiche la mémoire avant/après.

Exemple 2 — génération SHAP (explicabilité)
- Prompt utilisé :

```
Write a Python module to compute SHAP explanations for sklearn and tree-based models. Provide functions to: detect if shap is installed, compute shap values for a dataset, save summary (beeswarm) and waterfall plots to PNG, and fallback gracefully if shap is not installed.
```

- Résultat attendu : `src/shap_explanations.py` avec détection dynamique et fonctions `generate_shap_summary` et `plot_waterfall`.

Commentaires sur l'efficacité des prompts
- Prompts courts et ciblés donnent de bons squelettes de code; il faut ensuite itérer (exemples d'inputs/outputs, tailles de données) pour obtenir robustesse et gestion d'erreurs.

À compléter : ajoutez les prompts exacts et sorties textuelles (copier/coller) pour chaque interaction significative.
