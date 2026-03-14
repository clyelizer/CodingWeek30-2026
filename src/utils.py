# src/utils.py
"""
Fonctions utilitaires partagées dans le projet.
"""
import os
import random
import numpy as np

def set_seed(seed: int = 42):
    """Fixe les graines aléatoires pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

def ensure_dirs(*paths: str):
    """Crée les dossiers s'ils n'existent pas."""
    for p in paths:
        os.makedirs(p, exist_ok=True)

def print_section(title: str, width: int = 55):
    """Affiche un titre de section formaté."""
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")
