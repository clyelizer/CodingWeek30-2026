import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import joblib
import pandas as pd

d = joblib.load("data/processed/processed_data.joblib")
X_test  = d["X_test"]
y_test  = d["y_test"]

from src.train_model import load_model
model = load_model("models/random_forest.joblib")

# 5 positifs + 5 negatifs bien representes
pos  = X_test[y_test == 1].iloc[:5]
neg  = X_test[y_test == 0].iloc[:5]
sample   = pd.concat([pos, neg])
y_sample = y_test.loc[sample.index]

probas = model.predict_proba(sample)[:, 1]
preds  = (probas >= 0.5).astype(int)

print("=" * 90)
print(f"{'#':<4} {'Diag_reel':<12} {'Proba_RF':>10} {'Pred_RF':>9} {'Statut':<8}")
print("-" * 90)
for i, (idx, _) in enumerate(sample.iterrows()):
    real   = int(y_sample.loc[idx])
    p      = probas[i]
    pred   = int(p >= 0.5)
    statut = "✓ OK" if pred == real else "✗ ERREUR"
    print(f"{i+1:<4} {real:<12} {p:>10.3f} {pred:>9}   {statut}")

print()
print("=" * 90)
print("VALEURS CLINIQUES DES 10 CAS")
print("=" * 90)

cols_display = [
    "Lower_Right_Abd_Pain", "Migratory_Pain", "Nausea", "Ipsilateral_Rebound_Tenderness",
    "Body_Temperature", "WBC_Count", "Neutrophil_Percentage", "CRP",
    "Appendix_Diameter", "Age"
]
out = sample[cols_display].copy()
out.insert(0, "Diagnosis", y_sample.values)
out["Proba_RF"] = probas.round(3)
out["Pred"]     = preds
out.index       = range(1, 11)

# Affichage compact colonne par colonne
for col in out.columns:
    print(f"  {col:<35}: {list(out[col])}")
