from sklearn.utils import compute_class_weight


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
