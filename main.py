import config_loader
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# ✅ Vérifier que le GPU est bien utilisé
print("GPU Disponible:", torch.cuda.is_available())  
print("Nombre de GPUs:", torch.cuda.device_count())  

# ✅ Charger la config et se connecter à Hugging Face
CONFIG_PATH = "etc/config.yml"
config = config_loader.load_config(CONFIG_PATH)
login_token = config.get("HF_token", "")

if not login_token:
    raise ValueError("Le token de login est manquant dans le fichier de configuration.")
login(login_token)

# ✅ Charger le modèle original (avant quantization)
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Charger en FP16 pour économiser de la mémoire
model = AutoModelForCausalLM.from_pretrained(
    model_name, # ✅ FP16 pour éviter de saturer la VRAM
    device_map="auto"
)

# ✅ Texte d’entrée
input_text = "Pourquoi la quantification est-elle importante pour les LLM ?"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# ✅ Configuration de la quantization
#quant_config = BaseQuantizeConfig(
#    bits=8,            # 🔹 Essaie 16, 12, 10, 8, 6, 4 bits
#    group_size=128,    # 🔹 Taille de groupe (128 recommandé)
#    desc_act=False     # 🔹 Garde les activations en FP16/FP32 pour éviter de perdre trop en qualité
#)
#
# ✅ Quantifier le modèle
#quantized_model_dir = "./mistral_gptq_quantized"
#
#quantized_model = AutoGPTQForCausalLM.from_pretrained(
#    model_name, 
#    quantize_config=quant_config, 
#    device="cuda"
#)

#quantized_model.quantize_model()  # ⚠️ Étape obligatoire pour exécuter la quantization
#quantized_model.save_quantized(quantized_model_dir)  # Sauvegarde du modèle quantifié

# ✅ Test d'inférence après quantization
with torch.no_grad():  
    output = model.generate(**inputs, max_length=100)

# ✅ Affichage du texte généré
print("Texte généré :", tokenizer.decode(output[0], skip_special_tokens=True))
