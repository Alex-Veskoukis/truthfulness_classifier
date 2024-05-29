import optuna
from optuna.trial import TrialState
import torch
from transformers import TrainingArguments, Trainer, BertForSequenceClassification
from torch.utils.data import TensorDataset
from truthfulness_classifier.data_preprocessing import preprocess_data
from truthfulness_classifier.data_collator import CustomDataCollator
import logging
import os
import datetime
import yaml
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_path=os.path.join(os.path.dirname(__file__), '../config.yaml')):
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

def objective(trial, data_path, config):
    # Hyperparameters to optimize
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-5)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])

    # Data preprocessing
    train_encodings, test_encodings, y_train, y_test, tokenizer, label_to_id = preprocess_data(data_path, config)

    model = BertForSequenceClassification.from_pretrained(config['model']['name'], num_labels=len(label_to_id)).to(device)
    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], y_train)
    test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], y_test)

    # Create a unique output directory for each trial
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_number = trial.number
    output_dir = f'./results/optuna_trial_{trial_number}_{timestamp}'

    class_weights = get_class_weights(y_train.cpu().numpy(), len(label_to_id))

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=config['training']['warmup_steps'],
        weight_decay=config['training']['weight_decay'],
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_steps=500,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=CustomDataCollator(),
        class_weights=class_weights,
    )

    logger.info(f"Starting training with trial {trial_number}: learning_rate={learning_rate}, batch_size={batch_size}")
    trainer.train()
    logger.info(f"Training completed for trial {trial_number}")

    # Save the final model
    model_save_path = os.path.join(output_dir, 'final_model')
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    joblib.dump(label_to_id, os.path.join(model_save_path, 'label_to_id.pkl'))
    logger.info(f"Final model saved at {model_save_path}")

    eval_result = trainer.evaluate(eval_dataset=test_dataset)
    return eval_result['eval_loss'], model, tokenizer, label_to_id


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Hyperparameter tuning for Truthfulness Classifier")
    parser.add_argument('--data', type=str, required=True, help='Path to the training data CSV file')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')

    args = parser.parse_args()

    config = load_config(args.config)

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, args.data, config), n_trials=config['optuna']['n_trials'])

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    logger.info("Study statistics: ")
    logger.info(f"  Number of finished trials: {len(study.trials)}")
    logger.info(f"  Number of pruned trials: {len(pruned_trials)}")
    logger.info(f"  Number of complete trials: {len(complete_trials)}")

    logger.info("Best trial:")
    trial = study.best_trial

    logger.info(f"  Value: {trial.value}")

    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    # Train and save the best model
    best_loss, best_model, best_tokenizer, best_label_to_id = objective(trial, args.data, config)

    best_model_dir = './best_model'
    os.makedirs(best_model_dir, exist_ok=True)

    best_model.save_pretrained(best_model_dir)
    best_tokenizer.save_pretrained(best_model_dir)
    joblib.dump(best_label_to_id, os.path.join(best_model_dir, 'label_to_id.pkl'))

    logger.info(f"Best model saved to {best_model_dir}")

if __name__ == "__main__":
    main()
