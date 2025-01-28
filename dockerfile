# Utiliser l'image Mistral existante comme base
FROM ai/mistral:7B-Instruct-v0.2

# Installer pip si absent
RUN apt-get update && apt-get install -y python3-pip

# Installer les versions CPU de torch et torchvision
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Désactiver CUDA pour éviter toute recherche GPU
ENV CUDA_VISIBLE_DEVICES=-1

# Définir le répertoire de travail
WORKDIR /inteligentEnvironment-

# Copier tous les fichiers du projet dans le conteneur
COPY . .

# Définir la commande par défaut pour exécuter le script Python
CMD ["python3", "main.py"]
