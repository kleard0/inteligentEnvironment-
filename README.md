# Intelligent Environment

Ce projet vise à évaluer les performances de différents modèles de langage (LLMs) en utilisant des métriques telles que BLEU, ROUGE, perplexité, et BERTScore. Il permet également de comparer les temps de génération, la longueur des réponses, et la fréquence de génération des caractères.

## Structure du projet

```
├── .gitignore
├── config_loader.py
├── Etude sur la quantification de LLM.docx
├── idea
├── main.py
├── metrics.csv
├── model_installation.sh
├── README.md
├── __pycache__/
├── etc/
│   ├── config.template.yml
│   ├── config.yml
│   ├── environment.yml
```

### Description des fichiers

- **`main.py`** : Script principal qui charge la configuration, interroge les modèles via l'API Ollama, calcule les métriques, et sauvegarde les résultats dans un fichier CSV.
- **`config_loader.py`** : Module pour charger les fichiers de configuration YAML.
- **`model_installation.sh`** : Script Bash pour installer les modèles via la commande `ollama pull`.
- **`etc/config.yml`** : Fichier de configuration contenant les paramètres du projet, tels que l'URL de l'API, les modèles à tester, et le prompt.
- **`etc/environment.yml`** : Fichier de configuration pour créer un environnement Python avec les dépendances nécessaires.
- **`metrics.csv`** : Fichier généré contenant les résultats des évaluations des modèles.
- **`idea`** : Fichier texte contenant des idées pour optimiser les modèles.
- **`Etude sur la quantification de LLM.docx`** : Document d'étude sur la quantification des modèles de langage.

## Prérequis

- Python 3.10 ou supérieur
- [Ollama](https://ollama.com/) installé et configuré
- Environnement Python configuré avec les dépendances listées dans `etc/environment.yml`

## Installation

1. Clonez le dépôt :
   ```bash
   git clone <url-du-repo>
   cd intelligentEnvironment
   ```

2. Créez un environnement Python et installez les dépendances :
   ```bash
   conda env create -f etc/environment.yml
   conda activate intelligentEnvironment
   ```

3. Installez les modèles nécessaires :
   ```bash
   bash model_installation.sh
   ```

## Utilisation

1. Modifiez le fichier `etc/config.yml` pour personnaliser les modèles, le prompt, et d'autres paramètres.
2. Lancez le script principal pour évaluer les modèles :
   ```bash
   python main.py
   ```
3. Consultez les résultats dans le fichier `metrics.csv`.

## Fonctionnalités

- Évaluation des modèles avec les métriques BLEU, ROUGE, perplexité, et BERTScore.
- Mesure des temps de génération, des longueurs de réponse, et des fréquences de génération.
- Comparaison des performances de plusieurs modèles de langage.

## Idées d'amélioration

- Désactivation du calcul des gradients pour alléger les modèles.
- Quantization, pruning, et knowledge distillation pour optimiser les performances.

## Contributeurs

- **Kylian Leard**  
  Email : kylian.leardpro@gmail.com

## Licence

Ce projet est sous licence MIT. Consultez le fichier `LICENSE` pour plus d'informations.