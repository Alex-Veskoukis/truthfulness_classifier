import logging
import torch
from truthfulness_classifier.data_preprocessing import preprocess_single_statement
from truthfulness_classifier.utils import load_config, load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_truthfulness(statement_data, model_dir='./results', config_path='config.yaml'):
    config = load_config(config_path)
    model, tokenizer, label_to_id = load_model(model_dir)
    x_new = preprocess_single_statement(statement_data, tokenizer, max_length=config['model']['max_length'])

    model.eval()
    with torch.no_grad():
        outputs = model(x_new['input_ids'].to(device), x_new['attention_mask'].to(device))
        prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]

    id_to_label = {v: k for k, v in label_to_id.items()}
    prediction_label = id_to_label[prediction]
    confidence = torch.softmax(outputs.logits, dim=1).max().cpu().numpy()

    explanation = (
        f"The statement is predicted as {prediction_label} with a confidence score of {confidence:.2f}. "
        f"This is based on the content and context provided."
    )
    return prediction_label, explanation


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Predict the truthfulness of a statement")
    parser.add_argument('statement_data',
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

    with open(args.statement_data, 'r') as f:
        statement_data = json.load(f)

    prediction, explanation = predict_truthfulness(statement_data, model_dir=args.model_dir, config_path=args.config)
    print(f"Predicted label: {prediction}")
    print(explanation)


if __name__ == "__main__":
    main()
