"""
conftest.py — Configuration pytest pour le projet.
Ajoute la racine du projet dans sys.path pour que `from src.xxx import ...`
fonctionne depuis n'importe quel répertoire de lancement.
"""

import sys
import pathlib

# Racine du projet = dossier contenant ce fichier
_ROOT = pathlib.Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
