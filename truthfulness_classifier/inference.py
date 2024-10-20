import logging
import torch
from truthfulness_classifier.data_preprocessing import preprocess_single_statement
from truthfulness_classifier.utils import load_config, load_model
import warnings
import argparse
import json

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore", category=FutureWarning,
                        message="Passing the following arguments to `Accelerator` is deprecated")


def predict_truthfulness(statement_data, model_dir='./results', config_path='config.yaml'):
    config = load_config(config_path)
    try:
        model, tokenizer, label_to_id = load_model(model_dir)
    except Exception as e:
        logger.error(f"Failed to load model from {model_dir}: {e}")
        raise

    model.to(device)

    x_new = preprocess_single_statement(statement_data, tokenizer, max_length=config['model']['max_length'])

    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=x_new['input_ids'].to(device),
            attention_mask=x_new['attention_mask'].to(device)
        )
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).cpu().numpy()[0]

    id_to_label = {v: k for k, v in label_to_id.items()}
    prediction_label = id_to_label[prediction]
    confidence = torch.softmax(logits, dim=1).max().cpu().item()

    explanation = (
        f"The statement is predicted as {prediction_label} with a confidence score of {confidence:.2f}. "
        "This is based on the content and context provided."
    )
    return prediction_label, explanation



def main():
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Predict the truthfulness of a statement")
    parser.add_argument('statement_data_path',
                        type=str,
                        help='Path to the JSON file containing the statement data')
    parser.add_argument('--model_dir',
                        type=str,
                        default='./results',
                        help='Directory to load the model from')
    parser.add_argument('--config',
                        type=str,
                        default='config.yaml',
                        help='Path to the configuration file')

    args = parser.parse_args()

    with open(args.statement_data_path, 'r') as f:
        statement_data = json.load(f)

    prediction, explanation = predict_truthfulness(statement_data, model_dir=args.model_dir, config_path=args.config)
    print(f"Predicted label: {prediction}")
    print(explanation)


if __name__ == "__main__":
    main()
