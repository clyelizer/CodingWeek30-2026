# ─── PediAppendix — Dockerfile ───────────────────────────────────────────────
# Image légère Python 3.11 slim
# Commandes :
#   docker build -t pediappendix .
#   docker run -p 8000:8000 pediappendix

FROM python:3.11-slim

# Métadonnées
LABEL description="PediAppendix — Aide au diagnostic pédiatrique de l'appendicite"
LABEL version="2.0.0"

# Répertoire de travail dans le conteneur
WORKDIR /app

# Copie des dépendances en premier (pour le cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY src/       ./src/
COPY app/       ./app/
COPY models/    ./models/
COPY data/processed/ ./data/processed/
COPY conftest.py .

# Variable d'environnement : port d'écoute
ENV PORT=8000

# Exposition du port
EXPOSE 8000

# Démarrage de l'application
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
