import os
import torch
import joblib
import yaml
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer, BertForSequenceClassification
from truthfulness_classifier.data_preprocessing import preprocess_data, load_config

# Configure logging
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_dir):
    if not os.path.exists(model_dir):
        logger.error(f"The directory {model_dir} does not exist.")
        raise FileNotFoundError(f"The directory {model_dir} does not exist.")

    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(config_path):
        logger.error(f"The directory {model_dir} does not appear to have a file named config.json.")
        raise FileNotFoundError(f"The directory {model_dir} does not appear to have a file named config.json.")

    logger.info(f"Loading model from {model_dir}")
    model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    label_to_id = joblib.load(os.path.join(model_dir, 'label_to_id.pkl'))
    return model, tokenizer, label_to_id


def evaluate(model, tokenizer, label_to_id, data_path, config):
    logger.info(f"Evaluating model on {data_path}")

    # Preprocess data
    _, test_encodings, _, y_test, _, _ = preprocess_data(data_path, config)

    # Convert test encodings to DataLoader
    test_dataset = torch.utils.data.TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['training']['batch_size'])

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    return accuracy, precision, recall, f1


def update_config(best_model_dir, config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    config['model']['best_model_dir'] = best_model_dir

    with open(config_path, 'w') as file:
        yaml.safe_dump(config, file)
    logger.info(f"Updated config.yaml with the best model directory: {best_model_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate all models and update config with the best model")
    parser.add_argument('--data', type=str, required=True, help='Path to the test data CSV file')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing the result models')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')

    args = parser.parse_args()

    config = load_config(args.config)

    best_f1 = 0
    best_model_dir = None

    for model_dir in os.listdir(args.results_dir):
        full_model_dir = os.path.join(args.results_dir, model_dir)
        if os.path.isdir(full_model_dir):
            config_path = os.path.join(full_model_dir, 'config.json')
            if not os.path.exists(config_path):
                logger.error(f"Skipping {full_model_dir}: No config.json found")
                continue
            try:
                model, tokenizer, label_to_id = load_model(full_model_dir)
                _, _, _, f1 = evaluate(model, tokenizer, label_to_id, args.data, config)
                if f1 > best_f1:
                    best_f1 = f1
                    best_model_dir = full_model_dir
            except Exception as e:
                logger.error(f"Error evaluating model in {full_model_dir}: {e}")

    if best_model_dir:
        update_config(best_model_dir, args.config)
        logger.info(f"Best model is in {best_model_dir} with F1 score: {best_f1:.4f}")
    else:
        logger.info("No valid models found in the results directory.")
