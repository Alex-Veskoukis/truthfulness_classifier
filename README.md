# Truthfulness Classifier

This project aims to classify the truthfulness of statements using a multiclass classification approach with a TinyBERT model. The classifier can be trained, tuned using Optuna, and used for inference.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Predicting Truthfulness](#predicting-truthfulness)
  - [Hyperparameter Tuning with Optuna](#hyperparameter-tuning-with-optuna)
  - [Evaluating Models](#evaluating-models)
- [Project Structure](#project-structure)
- [Detailed Instructions](#Detailed Instructions)
- [Additional Resources](#Additional Resources)

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Alex-Veskoukis/truthfulness_classifier.git
cd truthfulness_classifier
```

### Step 2: Set Up a Conda Environment

Create and activate a conda environment for the project.

```bash
conda create -n Satalia_truthfulness_classifier python=3.9
conda activate Satalia_truthfulness_classifier
```

### Step 3: Install Dependencies

Install the necessary dependencies using \`requirements.txt\`.

```bash
pip install -r requirements.txt
```

### Step 4: Install the Package

Install the package in editable mode.

```bash
pip install -e .
```

## Usage

### Training the Model

To train the model, use the following command:

```bash
train-truthfulness --data <data_path> --config config.yaml
```
Example:
```bash
train-truthfulness --data tests/examples/data.csv --config config.yaml
```

- \`<data_path>\`: Path to the training data CSV file. (see tests/examples/data.csv)
- \`config.yaml\`: Path to the configuration file.

### Predicting Truthfulness

To predict the truthfulness of a statement, use the following command:

```bash
predict-truthfulness --statement_data <statement_data_path> --model_dir <model_directory> --config config.yaml
```

Example:
```bash
predict-truthfulness tests/examples/statement_data.json --model_dir results/model_20240529_233313 --config config.yaml
```

- \`<statement_data_path>\`: Path to the JSON file containing the statement data. (see tests/examples/statement_data.json)
- \`<model_directory>\`: Directory to load the model from. (defaults to './results')
- \`config.yaml\`: Path to the configuration file.

### Hyperparameter Tuning with Optuna

To perform hyperparameter tuning using Optuna, use the following command:

```bash
tune-truthfulness --data <data_path> --config config.yaml
```

Example:
```bash
tune-truthfulness tests/examples/data.csv --config config.yaml
```

- \`<data_path>\`: Path to the training data CSV file. (see tests/examples/data.csv)
- \`config.yaml\`: Path to the configuration file.

### Evaluating Models

To evaluate multiple models and determine the best one, use the following command:

```bash
python evaluate_all_models.py --data tests/examples/data.csv --results_dir ./results --config config.yaml
```

## Project Structure

```
pythonProject/
│
├── truthfulness_classifier/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── inference.py
│   ├── optuna_tuning.py
│
├── tests/
│   ├── examples/
│   │   ├── data.csv
│   │   └── statement_data.json
│   ├── test_data_preprocessing.py
│   ├── test_model_training.py
│   ├── test_inference.py
│
├── evaluate_all_models.py
├── evaluate_model.py
├── README.md
├── setup.py
├── requirements.txt
├── requirements_full.txt
├── config.yaml
├── logs/
├── results/
│   ├── model_20240529_233313/
```

## Detailed Instructions
### Data Preparation
Ensure your data is in a CSV format with the necessary columns: statement, subjects, speaker_name, speaker_job, statement_context, and Label.

Place your data file in the tests/examples/ directory for testing purposes or specify the path when running the scripts.

###  Config File
The config.yaml file should contain the necessary configuration parameters for the model training, such as:

```yaml
data:
  random_state: 42
  test_size: 0.2
model:
  best_model_dir: ./results/model_20240529_233313
  max_length: 512
  name: huawei-noah/TinyBERT_General_4L_312D
  num_labels: 6
optuna:
  n_trials: 5
training:
  batch_size: 16
  learning_rate: 2e-5
  logging_dir: ./logs
  num_epochs: 3
  output_dir: ./results
  warmup_steps: 500
  weight_decay: 0.01
```

### Preprocessing
To improve the preprocessing, the following techniques are employed:

Removing Stop Words: Common words that do not contribute to the model's understanding are removed.
Lemmatization: Reducing words to their base or root form to ensure consistency in word usage.
Spacy Model Loading: The Spacy model en_core_web_sm is used for advanced text preprocessing tasks.

### Downloading Necessary NLTK Data

Ensure you have the necessary NLTK data:

```pythonimport nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### Running Tests
Unit tests are provided to ensure the functionality of the data preprocessing, model training, and inference modules.

Run the tests using:

```bash
python -m unittest discover tests
```

## Additional Resources

The latest model can be downloaded from this [link](https://1drv.ms/u/s!AggiPSUtJpdntgAmQk3maPta9EHm?e=eXr5Bg) and
should be unpacked in the root directory.

The presentation can be found [here](https://1drv.ms/p/s!AggiPSUtJpdntgFMAzHmQg5Fz_Ze?e=APxmot).