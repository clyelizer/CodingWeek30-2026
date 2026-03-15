# src/utils.py
"""
Fonctions utilitaires partagées dans le projet.
"""
import os
import random
import numpy as np

def set_seed(seed: int = 42):
    """
    Fixe les graines aléatoires pour assurer la reproductibilité.

    Définit les graines pour `random`, `numpy` et `PYTHONHASHSEED`. Tente aussi
    d'initialiser PyTorch si présent. Utile pour runs reproductibles en ML.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

def ensure_dirs(*paths: str):
    """
    Crée les répertoires fournis s'ils n'existent pas.

    Wrapper pratique pour éviter de répéter `os.makedirs(..., exist_ok=True)`.
    Utilisé partout pour préparer les sorties et artefacts du pipeline.
    """
    for p in paths:
        os.makedirs(p, exist_ok=True)


def format_section(title: str, width: int = 55) -> str:
    """
    Construit et retourne une chaîne représentant un en-tête visuel.

    Utile pour les sorties console lisibles lors d'exécutions CLI.
    Retourne une chaîne contenant des lignes séparatrices autour du titre.
    """
    line = '─' * width
    return f"\n{line}\n  {title}\n{line}"

def print_section(title: str, width: int = 55):
    """
    Affiche un en-tête formaté sur la sortie standard.

    Simple utilitaire pour rendre la CLI plus lisible lors des étapes du pipeline.
    """
    print(format_section(title, width))
