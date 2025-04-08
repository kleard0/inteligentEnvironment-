import config_loader
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

print("GPU Disponible:", torch.cuda.is_available())  
print("Nombre de GPUs:", torch.cuda.device_count())  

CONFIG_PATH = "etc/config.yml"
config = config_loader.load_config(CONFIG_PATH)
login_token = config.get("HF_token", "")

if not login_token:
    raise ValueError("Le token de login est manquant dans le fichier de configuration.")
login(login_token)

model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)

input_text = "Pourquoi la quantification est-elle importante pour les LLM ?"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")


with torch.no_grad():  
    output = model.generate(**inputs, max_length=100)


print("Texte généré :", tokenizer.decode(output[0], skip_special_tokens=True))
