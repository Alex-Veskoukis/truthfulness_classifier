import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report
import joblib
import os
import logging
from truthfulness_classifier.data_preprocessing import preprocess_single_statement, preprocess_data
from truthfulness_classifier.utils import get_class_weights, load_config, CustomDataset
from transformers import DataCollatorWithPadding
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
warnings.filterwarnings("ignore", category=FutureWarning, module="accelerate.accelerator")
warnings.filterwarnings(
    "ignore",
    message=r".*Some weights of.*were not initialized.*",
    category=UserWarning,
    module="transformers.modeling_utils"
)
warnings.filterwarnings(
    "ignore",
    message=r".*Torch was not compiled with flash attention.*",
    category=UserWarning,
    module="transformers.models.bert.modeling_bert"
)
# Configure logging
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(device)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        if labels is not None:
            labels = labels.to(device)
            inputs["labels"] = labels
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
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
        self.output_dir = config['training']['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.label_to_id = None
        self.model = None  # Initialize the model as None; it will be loaded or created as needed

    def train(self, train_encodings, y_train, test_encodings, y_test, label_to_id):
        self.label_to_id = label_to_id

        # Create the datasets using the custom dataset class
        train_dataset = CustomDataset(train_encodings, y_train)
        test_dataset = CustomDataset(test_encodings, y_test)

        # Use DataCollatorWithPadding for dynamic padding
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Calculate class weights and move to device
        class_weights = get_class_weights(y_train.cpu().numpy(), self.num_labels).to(device)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.config['training']['num_epochs'],
            per_device_train_batch_size=self.config['training']['batch_size'],
            per_device_eval_batch_size=self.config['training']['batch_size'],
            gradient_accumulation_steps=4,
            warmup_steps=self.config['training']['warmup_steps'],
            weight_decay=self.config['training']['weight_decay'],
            logging_dir=self.config['training']['logging_dir'],
            logging_steps=100,
            eval_strategy="epoch",
            save_steps=500,
            save_total_limit=3,
            fp16=torch.cuda.is_available(),
            disable_tqdm=False
        )

        # Initialize the model
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels, ignore_mismatched_sizes=True).to(device)

        # Initialize the custom trainer
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

        # Evaluate the model
        predictions = trainer.predict(test_dataset)
        y_pred = torch.argmax(torch.tensor(predictions.predictions), dim=1).cpu().numpy()
        y_true = y_test.cpu().numpy()

        id_to_label = {v: k for k, v in label_to_id.items()}
        report = classification_report(
            y_true, y_pred, target_names=list(id_to_label.values()), zero_division=1)
        logger.info("Classification Report:\n" + report)

        # Save the trained model
        self.save_model()

    def save_model(self):
        logger.info(f"Saving model to {self.output_dir}")
        try:
            self.model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            joblib.dump(self.label_to_id, os.path.join(
                self.output_dir, 'label_to_id.pkl'))
            logger.info("Model saved successfully")
        except OSError as e:
            logger.error(f"OS error occurred while saving the model: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise

    def load_model(self, model_dir=None):
        model_dir = model_dir or self.output_dir
        logger.info(f"Loading model from {model_dir}")
        try:
            self.model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
            self.tokenizer = BertTokenizer.from_pretrained(model_dir)
            self.label_to_id = joblib.load(os.path.join(model_dir, 'label_to_id.pkl'))
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict(self, statement_data, tokenizer=None, max_length=None):
        if self.model is None:
            self.load_model()
        if tokenizer is None:
            tokenizer = self.tokenizer
        if max_length is None:
            max_length = self.config['model']['max_length']

        # Preprocess the input statement
        X_new = preprocess_single_statement(statement_data, tokenizer, max_length=max_length)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                X_new['input_ids'].to(device),
                attention_mask=X_new['attention_mask'].to(device)
            )
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).cpu().numpy()[0]

        id_to_label = {v: k for k, v in self.label_to_id.items()}
        prediction_label = id_to_label[prediction]
        confidence = torch.softmax(logits, dim=1).max().cpu().numpy()

        explanation = (
            f"The statement is predicted as {prediction_label} with a confidence score of {confidence:.2f}. "
            "This is based on the content and context provided."
        )
        return prediction_label, explanation


def main():
    import argparse
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)
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
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)
    main()
