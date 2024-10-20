import os
import logging
from collections import Counter

import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from imblearn.over_sampling import RandomOverSampler
from typing import Tuple, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

# Initialize stop words and lemmatizer
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

try:
    nltk.data.find('wordnet')
    lemmatizer = WordNetLemmatizer()
except LookupError:
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()

# Load Spacy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')


def preprocess_text(text: str) -> str:
    """
    Preprocess the input text by removing stop words and applying lemmatization.

    Args:
        text (str): The input text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    doc = nlp(text)
    tokens = [lemmatizer.lemmatize(token.lemma_) for token in doc if
              token.text.lower() not in stop_words and token.is_alpha]
    return ' '.join(tokens)


def preprocess_data(
        data_path: str,
        config: Dict[str, Any]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, BertTokenizer, Dict[str, int]]:
    """
    Preprocess the dataset for TinyBERT model training.

    Args:
        data_path (str): Path to the CSV data file.
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        Tuple containing training encodings, test encodings, training labels, test labels, tokenizer, and label mapping.
    """
    if not os.path.exists(data_path):
        logger.error(f"The file {data_path} does not exist.")
        raise FileNotFoundError(f"The file {data_path} does not exist.")

    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)

    # Check for required columns
    required_columns = [
        'statement', 'subjects', 'speaker_name', 'speaker_job',
        'speaker_state', 'speaker_affiliation', 'statement_context', 'Label'
    ]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"The following required columns are missing from the data: {missing_columns}")

    logger.info("Filling missing values and converting to strings")
    text_columns = [
        'statement', 'subjects', 'speaker_name', 'speaker_job',
        'speaker_state', 'speaker_affiliation', 'statement_context'
    ]
    for col in text_columns:
        data[col] = data[col].fillna('').astype(str)

    logger.info("Combining text fields for tokenization")
    data['combined_text'] = data.apply(
        lambda row: f"{row['statement']} {row['subjects']} {row['speaker_name']} {row['speaker_job']} "
                    f"{row['speaker_state']} {row['speaker_affiliation']} {row['statement_context']}",
        axis=1
    )

    logger.info("Applying text preprocessing")
    data['processed_text'] = data['combined_text'].apply(preprocess_text)

    X = data['processed_text'].tolist()
    y = data['Label'].tolist()

    logger.info("Creating label to ID mapping")
    unique_labels = sorted(set(y))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    labels = [label_to_id[label] for label in y]

    logger.info(f"Original class distribution: {Counter(labels)}")

    logger.info("Handling class imbalance using RandomOverSampler")
    ros = RandomOverSampler(random_state=config['data']['random_state'])
    # Reshape X to 2D array for RandomOverSampler
    X = np.array(X).reshape(-1, 1)
    X_resampled, labels_resampled = ros.fit_resample(X, labels)
    X_resampled = X_resampled.flatten()

    logger.info(f"Resampled class distribution: {Counter(labels_resampled)}")

    logger.info(f"Using tokenizer: {config['model']['name']}")
    tokenizer = BertTokenizer.from_pretrained(config['model']['name'])

    logger.info("Tokenizing input texts")
    encodings = tokenizer(
        list(X_resampled),
        add_special_tokens=True,
        max_length=config['model']['max_length'],
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids_resampled = encodings['input_ids']
    attention_masks_resampled = encodings['attention_mask']
    labels_resampled = torch.tensor(labels_resampled)

    logger.info(f"Splitting data into training and test sets with test size {config['data']['test_size']}")
    input_ids_train, input_ids_test, attention_mask_train, attention_mask_test, y_train, y_test = train_test_split(
        input_ids_resampled,
        attention_masks_resampled,
        labels_resampled,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )

    train_encodings = {'input_ids': input_ids_train, 'attention_mask': attention_mask_train}
    test_encodings = {'input_ids': input_ids_test, 'attention_mask': attention_mask_test}

    logger.info("Preprocessing complete")
    return train_encodings, test_encodings, y_train, y_test, tokenizer, label_to_id


def preprocess_test_data(
        data_path: str,
        tokenizer,
        label_to_id: Dict[str, int],
        config: Dict[str, Any]
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Preprocess the test data.

    Args:
        data_path (str): Path to the test data CSV file.
        tokenizer: The tokenizer used for preprocessing.
        label_to_id (Dict[str, int]): Mapping from labels to IDs.
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        Tuple containing test encodings and test labels.
    """
    if not os.path.exists(data_path):
        logger.error(f"The file {data_path} does not exist.")
        raise FileNotFoundError(f"The file {data_path} does not exist.")

    logger.info(f"Loading test data from {data_path}")
    data = pd.read_csv(data_path)

    # Ensure required columns are present
    required_columns = [
        'statement', 'subjects', 'speaker_name', 'speaker_job',
        'speaker_state', 'speaker_affiliation', 'statement_context', 'Label'
    ]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"The following required columns are missing from the data: {missing_columns}")

    # Preprocess text data
    # (Assuming you have a function to preprocess the text data)
    data['processed_text'] = data.apply(lambda row: preprocess_text(
        f"{row['statement']} {row['subjects']} {row['speaker_name']} {row['speaker_job']} "
        f"{row['speaker_state']} {row['speaker_affiliation']} {row['statement_context']}"
    ), axis=1)

    X_test = data['processed_text'].tolist()
    y_test = data['Label'].apply(lambda x: label_to_id.get(x, -1)).tolist()

    # Remove samples with unknown labels
    valid_indices = [i for i, label in enumerate(y_test) if label != -1]
    X_test = [X_test[i] for i in valid_indices]
    y_test = [y_test[i] for i in valid_indices]

    logger.info("Tokenizing test data")
    encodings = tokenizer(
        X_test,
        add_special_tokens=True,
        max_length=config['model']['max_length'],
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    y_test = torch.tensor(y_test, dtype=torch.long)

    test_encodings = {'input_ids': encodings['input_ids'], 'attention_mask': encodings['attention_mask']}

    return test_encodings, y_test


def preprocess_single_statement(
        statement_data: Dict[str, Any],
        tokenizer: BertTokenizer,
        max_length: int
) -> Dict[str, torch.Tensor]:
    """
    Preprocess a single statement for prediction with a TinyBERT model.

    Args:
        statement_data (Dict[str, Any]): The statement data to preprocess.
        tokenizer (BertTokenizer): The tokenizer to use.
        max_length (int): The maximum sequence length.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing 'input_ids' and 'attention_mask'.
    """
    statement_data = {key: (value if value is not None else '') for key, value in statement_data.items()}
    combined_text = (
        f"{statement_data['statement']} {statement_data['subjects']} {statement_data['speaker_name']} "
        f"{statement_data['speaker_job']} {statement_data['statement_context']}"
    )

    combined_text = preprocess_text(combined_text)

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
