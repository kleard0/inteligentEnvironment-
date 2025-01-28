from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_name = "mistralai/mistral-7b"  # Nom du modèle

    # Chargement tokenizer + modèle
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=None # desactive utilisation GPU
    )
    model = model.to("cpu")  # Force utilisation CPU

    # exemple de prompt
    prompt = "Comment vas-tu ?"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=100)

    # Afficher la réponse générée
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()


