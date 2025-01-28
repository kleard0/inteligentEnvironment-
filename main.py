from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_name = "mistralai/mistral-7b"  # Nom du modèle

    # Charger le tokenizer et le modèle
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=None  # Désactiver la gestion automatique des GPU
    )
    model = model.to("cpu")  # Forcer le modèle à s'exécuter sur CPU

    # Exemple d'entrée utilisateur
    prompt = "Bonjour, peux-tu m'expliquer la théorie de la relativité ?"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=100)

    # Afficher la réponse générée
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_name = "mistralai/mistral-7b"  # Nom du modèle

    # Charger le tokenizer et le modèle
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=None  # Désactiver la gestion automatique des GPU
    )
    model = model.to("cpu")  # Forcer le modèle à s'exécuter sur CPU

    # Exemple d'entrée utilisateur
    prompt = "Bonjour, peux-tu m'expliquer la théorie de la relativité ?"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=100)

    # Afficher la réponse générée
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
