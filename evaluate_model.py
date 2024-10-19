import os
import torch
import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer, BertForSequenceClassification
from truthfulness_classifier.data_preprocessing import preprocess_data
from truthfulness_classifier.utils import load_config

# Configure logging
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_dir):
    if not os.path.exists(model_dir):
        logger.error(f"The directory {model_dir} does not exist.")
        raise FileNotFoundError(f"The directory {model_dir} does not exist.")

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the Truthfulness Classifier")
    parser.add_argument('--data', type=str, required=True, help='Path to the test data CSV file')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory to load the model from')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')

    args = parser.parse_args()

    config = load_config(args.config)

    model, tokenizer, label_to_id = load_model(args.model_dir)
    evaluate(model, tokenizer, label_to_id, args.data, config)
