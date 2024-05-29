from truthfulness_classifier.data_preprocessing import preprocess_data
from truthfulness_classifier.model_training import TruthfulnessClassifier

# Preprocess the data
train_encodings, test_encodings, y_train, y_test, tokenizer, label_to_id = preprocess_data("path/to/data.csv")

# Initialize and train the classifier
classifier = TruthfulnessClassifier(num_labels=len(label_to_id))
classifier.train(train_encodings, y_train, test_encodings, y_test)


import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib

# Define paths
model_dir = "./results/model_20240528_214408"
tokenizer_name = 'huawei-noah/TinyBERT_General_4L_312D'
label_to_id_path = os.path.join(model_dir, 'label_to_id.pkl')

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained(model_dir).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
label_to_id = joblib.load(label_to_id_path)

# Save the tokenizer to the same directory
tokenizer.save_pretrained(model_dir)
