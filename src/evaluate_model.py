# src/evaluate_model.py
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    classification_report, RocCurveDisplay
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import os

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name='Model', save_roc=False, output_dir='reports/figures'):
    """
    Entraîne le modèle, calcule les métriques sur le test set et effectue une validation croisée.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'ROC-AUC': roc_auc_score(y_test, y_proba),
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-score': f1_score(y_test, y_pred, zero_division=0),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    metrics['CV ROC-AUC mean'] = cv_scores.mean()
    metrics['CV ROC-AUC std'] = cv_scores.std()

    # Affichage
    print(f"\n{'─'*45}")
    print(f"  {model_name}")
    print(f"{'─'*45}")
    for k, v in metrics.items():
        print(f"  {k:<22}: {v:.4f}")
    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, y_pred))
    print("\n" + classification_report(y_test, y_pred, zero_division=0))

    # Courbe ROC (optionnelle)
    if save_roc:
        os.makedirs(output_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6,5))
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name=model_name)
        ax.set_title(f"ROC Curve - {model_name}")
        path = os.path.join(output_dir, f"roc_{model_name.replace(' ', '_')}.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"ROC curve saved → {path}")

    return metrics