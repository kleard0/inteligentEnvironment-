"""
20/04/2025
author : Kylian Leard 
kylian.leardpro@gmail.com
"""

import config_loader
import requests
from nltk.translate.bleu_score import sentence_bleu
import evaluate
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import pandas as pd
import bert_score
import time

# Chargement de la configuration
CONFIG_PATH = "etc/config.yml"
config = config_loader.load_config(CONFIG_PATH)
login_token = config.get("HF_token", "")
# Configuration
OLLAMA_URL = config["OLLAMA_URL"]
MODEL_NAMES = config["MODEL_NAME"]  
PROMPT =  config["PROMPT"]
REFERENCE = config["REFERENCE"]

# Variables pour stocker les score
bleu_list = []
rouge_list = []
perplexity_list = []
precision_list =  []
recall_list = []
f1_list = []

# Charger le modèle de perplexité GPT2 
tokenizer_perplexity = GPT2Tokenizer.from_pretrained("gpt2")
model_perplexity = GPT2LMHeadModel.from_pretrained("gpt2")

# Charger la métrique ROUGE de HuggingFace
rouge = evaluate.load("rouge")

generation_times = []
response_lengths = []
generation_frequencies = []
response_list = []
for model_name in MODEL_NAMES:
    payload = {
        "model": model_name,
        "prompt": PROMPT,
        "stream": False
    }

    try:
        start_time = time.time()
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        end_time = time.time()

        data = response.json()
        generation_time = end_time - start_time
        response_text = data["response"]
        response_length = len(response_text)
        generation_frequency = response_length / generation_time if generation_time > 0 else 0

        print(f"Nom du modèle : {model_name}")
        print(f"Réponse du modèle : {response_text}")

        # BLEU
        bleu_score = sentence_bleu([REFERENCE], response_text)

        # ROUGE
        rouge_score = rouge.compute(predictions=[response_text], references=[REFERENCE])

        # Perplexité
        inputs = tokenizer_perplexity(response_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model_perplexity(**inputs, labels=inputs["input_ids"])
            log_likelihood = outputs.loss.item()
        perplexity_score = torch.exp(torch.tensor(log_likelihood))

        # BERTScore
        P, R, F1 = bert_score.score([response_text], [REFERENCE], lang="fr")

        # Ajout des scores aux listes
        precision_list.append(P[0])
        recall_list.append(R[0])
        f1_list.append(F1[0])
        perplexity_list.append(perplexity_score.item())
        bleu_list.append(bleu_score)
        rouge_list.append(rouge_score["rouge1"])

        # Ajout des mesures temporelles
        generation_times.append(generation_time)
        response_lengths.append(response_length)
        generation_frequencies.append(generation_frequency)
        response_list.append(response_text)

    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la requête à Ollama : {e}")

# Ajout des nouvelles colonnes dans le DataFrame
metrics_df = pd.DataFrame({
    "model_name": MODEL_NAMES,
    "bleu_score": bleu_list,
    "rouge_score": rouge_list,
    "perplexity_score": perplexity_list,
    "precision": precision_list,
    "recall": recall_list,
    "f1_score": f1_list,
    "generation_time_sec": generation_times,
    "response_length_chars": response_lengths,
    "generation_freq_chars_per_sec": generation_frequencies,
    "response": response_list,
})

# Sauvegarde CSV
metrics_df.to_csv("metrics.csv", index=False)