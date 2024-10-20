import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from truthfulness_classifier.data_preprocessing import preprocess_test_data
from truthfulness_classifier.utils import load_config, load_model
from truthfulness_classifier.utils import CustomDataset  # Assuming this module exists
from typing import Tuple, Dict, Any
import logging
import warnings

# Suppress specific warnings if necessary
warnings.filterwarnings(
    "ignore",
    message=r".*Some weights of.*were not initialized.*",
    category=UserWarning,
    module="transformers.modeling_utils"
)

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(
        model: torch.nn.Module,
        tokenizer,
        label_to_id: Dict[str, int],
        data_path: str,
        config: Dict[str, Any]
) -> Tuple[float, float, float, float]:
    """
    Evaluate the model on the test data.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        tokenizer: The tokenizer used for preprocessing.
        label_to_id (Dict[str, int]): Mapping from labels to IDs.
        data_path (str): Path to the test data CSV file.
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        Tuple[float, float, float, float]: Accuracy, precision, recall, and F1 score.
    """
    logger.info(f"Evaluating model on {data_path}")

    # Preprocess test data
    test_encodings, y_test = preprocess_test_data(data_path, tokenizer, label_to_id, config)

    # Create the test dataset
    test_dataset = CustomDataset(test_encodings, y_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )

    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            # Move tensors to the correct device
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch['labels'].cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    return accuracy, precision, recall, f1


if __name__ == "__main__":
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the Truthfulness Classifier")
    parser.add_argument('--data', type=str, required=True, help='Path to the test data CSV file')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory to load the model from')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    config = load_config(args.config)

    try:
        model, tokenizer, label_to_id = load_model(args.model_dir)
    except Exception as e:
        logger.error(f"Failed to load model from {args.model_dir}: {e}")
        exit(1)

    evaluate(model, tokenizer, label_to_id, args.data, config)
