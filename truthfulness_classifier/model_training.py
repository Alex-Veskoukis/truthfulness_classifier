import torch
from torch.utils.data import TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report
import joblib
import os
import logging
import datetime
import yaml
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from truthfulness_classifier.data_preprocessing import preprocess_single_statement, preprocess_data
from truthfulness_classifier.data_collator import CustomDataCollator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_class_weights(labels, num_classes):
    class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=labels)
    return torch.tensor(class_weights, dtype=torch.float).to(device)

class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

class TruthfulnessClassifier:
    def __init__(self, config):
        self.config = config
        self.model_name = config['model']['name']
        self.num_labels = config['model']['num_labels']
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(config['training']['output_dir'], f"model_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels).to(device)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.label_to_id = None

    def train(self, train_encodings, y_train, test_encodings, y_test, label_to_id):
        self.label_to_id = label_to_id
        train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], y_train)
        test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], y_test)

        data_collator = CustomDataCollator()
        class_weights = get_class_weights(y_train.cpu().numpy(), self.num_labels)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.config['training']['num_epochs'],
            per_device_train_batch_size=self.config['training']['batch_size'],
            per_device_eval_batch_size=self.config['training']['batch_size'],
            gradient_accumulation_steps=4,
            warmup_steps=self.config['training']['warmup_steps'],
            weight_decay=self.config['training']['weight_decay'],
            logging_dir=self.config['training']['logging_dir'],
            logging_steps=10,
            evaluation_strategy="epoch",
            save_steps=500,
            save_total_limit=3,
            fp16=torch.cuda.is_available(),
        )

        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            class_weights=class_weights,
        )

        logger.info("Starting training")
        trainer.train()
        logger.info("Training complete")

        predictions = trainer.predict(test_dataset)
        y_pred = torch.argmax(torch.tensor(predictions.predictions), dim=1).numpy()
        y_true = y_test.numpy()

        id_to_label = {v: k for k, v in label_to_id.items()}
        report = classification_report(y_true, y_pred, target_names=list(id_to_label.values()), zero_division=1)
        logger.info("Classification Report:\n" + report)

        self.save_model()

    def save_model(self):
        logger.info(f"Saving model to {self.output_dir}")
        try:
            self.model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            joblib.dump(self.label_to_id, os.path.join(self.output_dir, 'label_to_id.pkl'))
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def load_model(self):
        logger.info(f"Loading model from {self.output_dir}")
        try:
            self.model = BertForSequenceClassification.from_pretrained(self.output_dir).to(device)
            self.tokenizer = BertTokenizer.from_pretrained(self.output_dir)
            self.label_to_id = joblib.load(os.path.join(self.output_dir, 'label_to_id.pkl'))
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def predict(self, statement_data, tokenizer=None, max_length=None):
        self.load_model()
        if tokenizer is None:
            tokenizer = self.tokenizer
        if max_length is None:
            max_length = self.config['model']['max_length']
        X_new = preprocess_single_statement(statement_data, tokenizer, max_length=max_length)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_new['input_ids'].to(device), X_new['attention_mask'].to(device))
            prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]

        id_to_label = {v: k for k, v in self.label_to_id.items()}
        prediction_label = id_to_label[prediction]
        confidence = torch.softmax(outputs.logits, dim=1).max().cpu().numpy()

        explanation = f"The statement is predicted as {prediction_label} with a confidence score of {confidence:.2f}. This is based on the content and context provided."
        return prediction_label, explanation

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train the Truthfulness Classifier")
    parser.add_argument('--data', type=str, required=True, help='Path to the training data CSV file')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')

    args = parser.parse_args()

    config = load_config(args.config)

    if not os.path.exists(config['training']['output_dir']):
        os.makedirs(config['training']['output_dir'])

    logger.info(f"Output directory: {config['training']['output_dir']}")

    train_encodings, test_encodings, y_train, y_test, tokenizer, label_to_id = preprocess_data(args.data, config)
    classifier = TruthfulnessClassifier(config)
    classifier.train(train_encodings, y_train, test_encodings, y_test, label_to_id)

if __name__ == "__main__":
    main()
