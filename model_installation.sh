#!/bin/bash

MODELS=(
  "mistral:7b-instruct-fp16"
  "mistral:7b-instruct-v0.2-fp16"
  "mistral:7b-instruct-q8_0"
  "mistral:7b-instruct-v0.2-q8_0"
  "mistral:7b-instruct-q6_K"
  "mistral:7b-instruct-v0.2-q6_K"
  "mistral:7b-instruct-q5_1"
  "mistral:7b-instruct-v0.2-q5_1"
  "mistral:7b-instruct-q5_0"
  "mistral:7b-instruct-v0.2-q5_0"
  "mistral:7b-instruct-q5_K_M"
  "mistral:7b-instruct-v0.2-q5_K_M"
  "mistral:7b-instruct-q5_K_S"
  "mistral:7b-instruct-v0.2-q5_K_S"
  "mistral:7b-instruct-q4_1"
  "mistral:7b-instruct-v0.2-q4_1"
  "mistral:7b-instruct-q4_0"
  "mistral:7b-instruct-v0.2-q4_0"
  "mistral:7b-instruct-q4_K_M"
  "mistral:7b-instruct-v0.2-q4_K_M"
  "mistral:7b-instruct-q4_K_S"
  "mistral:7b-instruct-v0.2-q4_K_S"
  "mistral:7b-instruct-q3_K_L"
  "mistral:7b-instruct-v0.2-q3_K_L"
  "mistral:7b-instruct-q3_K_M"
  "mistral:7b-instruct-v0.2-q3_K_M"
  "mistral:7b-instruct-q3_K_S"
  "mistral:7b-instruct-v0.2-q3_K_S"
  "mistral:7b-instruct-q2_K"
  "mistral:7b-instruct-v0.2-q2_K"
)

# Installer chaque modèle avec ollama pull
for model in "${MODELS[@]}"; do
  echo "Installation du modèle: $model"
  ollama pull "$model"
done

echo "Tous les modèles ont été installés."
