from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import torch
from huggingface_hub import login
# AutoModelForCausakLM est une classe permettant de charger un LLM
# AutoTokenizer est une classe permetant de convertir du texte en tokens (format compréhensible
# par le modèle
import torch
print(torch.cuda.is_available())  # Doit afficher True
print(torch.cuda.device_count())  # Doit afficher un nombre > 0 si un GPU est détecté

login("")
model_name = "mistralai/Mistral-7B-v0.1" #nom du modèle à charger (huggingface)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Le tokenizer est responsable de la conversion du texte en tokens (nombres) et inversement.

logging.get_logger("transformers").setLevel(logging.INFO)
# Configuration du niveau de log pour afficher les messages d'information liés aux transformers

input_text = "Pourquoi la quantification est-elle importante pour les LLM ?"
# text d'entrée pour le modèle

inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

with torch.no_grad():
# Désactivation du calcul des gradients pour économiser de la mémoire et accélérer l'inférence
    output = model.generate(**inputs, max_length=100)
    # Génération de texte par le modèle avec une limite de 100 tokenss
print(tokenizer.decode(output[0], skip_special_tokens=True))
# Décodage du texte généré en supprimant les tokens spéciaux