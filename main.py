import config_loader
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from huggingface_hub import login

# Vérifie l'accès GPU
print("GPU Disponible:", torch.cuda.is_available())  
print("Nombre de GPUs:", torch.cuda.device_count())  

# Chargement de la configuration
CONFIG_PATH = "etc/config.yml"
config = config_loader.load_config(CONFIG_PATH)
login_token = config.get("HF_token", "")
login(login_token)

# Définition du modèle et du prompt
model_name = config.get("GPTQ-4bit", "")
input_text = "Pourquoi la quantification est-elle importante pour les LLM ?"

# Chargement du tokenizer (une seule fois)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Détection automatique du device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fonction de génération de texte
def answer_generation(prompt: str, model, tokenizer): 
    """
    Génère une réponse à partir d'un prompt et d'un modèle LLM.

    :param prompt: Texte d'entrée
    :param model: Modèle LLM déjà chargé
    :param tokenizer: Tokenizer associé au modèle
    :return: Réponse générée
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_length=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Chargement du modèle normal
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
print("\nRéponse avec le modèle quantifié GPTQ 8 bits :")
print(answer_generation(input_text, model, tokenizer))
