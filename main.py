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

# Détection du device (CUDA si dispo, sinon CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Utilisation du device : {device}")

# Chargement de la configuration
CONFIG_PATH = "etc/config.yml"
config = config_loader.load_config(CONFIG_PATH)
login_token = config.get("HF_token", "")
OLLAMA_URL = config["OLLAMA_URL"]
MODEL_NAMES = config["MODEL_NAME"]  
PROMPT = config["PROMPT"]
REFERENCE = config["REFERENCE"]

# Variables pour stocker les scores
bleu_list = []
rouge_list = []
perplexity_list = []
precision_list = []
recall_list = []
f1_list = []

# Chargement modèle et tokenizer pour la perplexité
tokenizer_perplexity = GPT2Tokenizer.from_pretrained("gpt2")
model_perplexity = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
model_perplexity.eval()

# ROUGE via HuggingFace
rouge = evaluate.load("rouge")

# Pour les statistiques de performance
generation_times = []
response_lengths = []
generation_frequencies = []
response_list = []
gpu_times = []
gpu_memories = []

for model_name in MODEL_NAMES:
    payload = {
        "model": model_name,
        "prompt": PROMPT,
        "stream": False
    }

    try:
        # Mesure du temps CPU
        start_time = time.time()

        # Temps GPU si CUDA dispo
        if torch.cuda.is_available():
            start_gpu_time = torch.cuda.Event(enable_timing=True)
            end_gpu_time = torch.cuda.Event(enable_timing=True)
            start_gpu_time.record()

        # Appel du modèle
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()

        if torch.cuda.is_available():
            end_gpu_time.record()
            torch.cuda.synchronize()
            gpu_time = start_gpu_time.elapsed_time(end_gpu_time)
            memory_used = torch.cuda.memory_allocated()
        else:
            gpu_time = 0
            memory_used = 0

        end_time = time.time()
        generation_time = end_time - start_time

        data = response.json()
        response_text = data["response"]
        response_length = len(response_text)
        generation_frequency = response_length / generation_time if generation_time > 0 else 0

        print(f"\n--- Résultat pour {model_name} ---")
        print(f"Réponse : {response_text}")
        print(f"Temps GPU : {gpu_time:.2f} ms | Mémoire GPU : {memory_used} octets")

        # BLEU
        bleu_score = sentence_bleu([REFERENCE], response_text)

        # ROUGE
        rouge_score = rouge.compute(predictions=[response_text], references=[REFERENCE])

        # Perplexité
        inputs = tokenizer_perplexity(response_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model_perplexity(**inputs, labels=inputs["input_ids"])
            log_likelihood = outputs.loss.item()
        perplexity_score = torch.exp(torch.tensor(log_likelihood))

        # BERTScore
        P, R, F1 = bert_score.score([response_text], [REFERENCE], lang="fr")

        # Ajout des scores
        precision_list.append(P[0].item())
        recall_list.append(R[0].item())
        f1_list.append(F1[0].item())
        perplexity_list.append(perplexity_score.item())
        bleu_list.append(bleu_score)
        rouge_list.append(rouge_score["rouge1"])

        generation_times.append(generation_time)
        response_lengths.append(response_length)
        generation_frequencies.append(generation_frequency)
        response_list.append(response_text)
        gpu_times.append(gpu_time)
        gpu_memories.append(memory_used)

    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la requête à Ollama : {e}")
    except Exception as e:
        print(f"Erreur durant le traitement du modèle {model_name} : {e}")

# Création du DataFrame final
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
    "gpu_time_ms": gpu_times,
    "gpu_memory_bytes": gpu_memories,
    "response": response_list,
})

# Sauvegarde CSV
metrics_df.to_csv("metrics_with_gpu.csv", index=False)
print("\n✅ Fichier CSV 'metrics_with_gpu.csv' généré avec succès.")
