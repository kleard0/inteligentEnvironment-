
from transformers import AutoModelForCausalLM, AutoTokenizer

# Remplacez 'mistralai/Mistral-7B' par le nom exact du modèle sur Hugging Face
model_name = "mistralai/Mistral-7B-v0.1"


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# Exemple de génération de texte
prompt = "Comment fonctionne Mistral 7B ?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(inputs.input_ids, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


