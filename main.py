import config_loader
import requests

# Chargement de la configuration
CONFIG_PATH = "etc/config.yml"
config = config_loader.load_config(CONFIG_PATH)
login_token = config.get("HF_token", "")
# Configuration
OLLAMA_URL = config["OLLAMA_URL"]
MODEL_NAME = config["MODEL_NAME"]
PROMPT =  config["PROMPT"]

payload = {
    "model": MODEL_NAME,
    "prompt": PROMPT,
    "stream": False  
}

try:
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    data = response.json()
    print(" Réponse du modèle :", data["response"])

except requests.exceptions.RequestException as e:
    print(" Erreur lors de la requête à Ollama :", e)
