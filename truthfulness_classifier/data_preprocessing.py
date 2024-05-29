import os
import logging
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path=os.path.join(os.path.dirname(__file__), '../config.yaml')):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def preprocess_data(data_path, config):
    """
    Preprocess the dataset for TinyBERT model training.
    """
    if not os.path.exists(data_path):
        logger.error(f"The file {data_path} does not exist.")
        raise FileNotFoundError(f"The file {data_path} does not exist.")

    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)

    logger.info("Filling missing values")
    data['speaker_job'] = data['speaker_job'].fillna('')
    data['speaker_state'] = data['speaker_state'].fillna('')
    data['statement_context'] = data['statement_context'].fillna('')

    logger.info("Combining text fields for tokenization")
    data['combined_text'] = data.apply(
        lambda row: f"{row['statement']} {row['subjects']} {row['speaker_name']} {row['speaker_job']} {row['statement_context']}",
        axis=1
    )

    X = data['combined_text'].tolist()
    y = data['Label'].tolist()

    logger.info("Creating label to ID mapping")
    unique_labels = sorted(data['Label'].unique())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}

    logger.info(f"Using tokenizer: {config['model']['name']}")
    tokenizer = BertTokenizer.from_pretrained(config['model']['name'])

    logger.info("Tokenizing input texts")
    encodings = tokenizer(X, add_special_tokens=True, max_length=config['model']['max_length'], padding='max_length', truncation=True, return_tensors='pt')

    input_ids = encodings['input_ids']
    attention_masks = encodings['attention_mask']
    labels = torch.tensor([label_to_id[label] for label in y])

    logger.info(f"Splitting data into training and test sets with test size {config['data']['test_size']}")
    input_ids_train, input_ids_test, attention_mask_train, attention_mask_test, y_train, y_test = train_test_split(
        input_ids, attention_masks, labels, test_size=config['data']['test_size'], random_state=config['data']['random_state'])

    train_encodings = {'input_ids': input_ids_train, 'attention_mask': attention_mask_train}
    test_encodings = {'input_ids': input_ids_test, 'attention_mask': attention_mask_test}

    logger.info("Preprocessing complete")
    return train_encodings, test_encodings, y_train, y_test, tokenizer, label_to_id

def preprocess_single_statement(statement_data, tokenizer, max_length):
    """
    Preprocess a single statement for prediction with a TinyBERT model.
    """
    statement_data = {key: (value if value is not None else '') for key, value in statement_data.items()}
    combined_text = f"{statement_data['statement']} {statement_data['subjects']} {statement_data['speaker_name']} {statement_data['speaker_job']} {statement_data['statement_context']}"

    encoded_dict = tokenizer.encode_plus(
        combined_text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']

    return {'input_ids': input_ids, 'attention_mask': attention_mask}
