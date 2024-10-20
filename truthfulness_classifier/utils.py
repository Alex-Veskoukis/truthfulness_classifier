import joblib
import yaml
from sklearn.utils import compute_class_weight
import torch
import os
import numpy as np
import logging
import torch
from transformers import DefaultDataCollator

from transformers import BertForSequenceClassification, BertTokenizer

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path=os.path.join(os.path.dirname(__file__), '../config.yaml')):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_class_weights(labels, num_classes):
    class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=labels)
    return torch.tensor(class_weights, dtype=torch.float).to(device)


def load_model(model_dir='./results'):
    if not os.path.exists(model_dir):
        logger.error(f"The directory {model_dir} does not exist.")
        raise FileNotFoundError(f"The directory {model_dir} does not exist.")

    logger.info(f"Loading model from {model_dir}")
    model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    label_to_id = joblib.load(os.path.join(model_dir, 'label_to_id.pkl'))
    return model, tokenizer, label_to_id


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings  # Dictionary of tensors
        self.labels = labels  # Tensor of labels

    def __getitem__(self, idx):
        # Return a dictionary with keys: 'input_ids', 'attention_mask', 'labels'
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


class CustomDataCollator(DefaultDataCollator):
    def __call__(self, features, **kwargs):
        if isinstance(features[0], dict):
            batch = {key: torch.stack([f[key] for f in features]) for key in features[0]}
        else:
            # Handling case where features are tuples
            batch = {
                "input_ids": torch.stack([f[0] for f in features]),
                "attention_mask": torch.stack([f[1] for f in features]),
                "labels": torch.stack([f[2] for f in features])
            }
        return batch
