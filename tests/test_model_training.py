import unittest
import os
import yaml
from truthfulness_classifier.model_training import TruthfulnessClassifier, load_config
from truthfulness_classifier.data_preprocessing import preprocess_data
import torch


class TestModelTraining(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.data_path = 'tests/examples/data.csv'
        cls.model_dir = 'tests/model_20240528_214408'
        os.makedirs(cls.model_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        os.rmdir(cls.model_dir)

    def test_training(self):
        train_encodings, test_encodings, y_train, y_test, tokenizer, label_to_id = preprocess_data(self.data_path,
                                                                                                   self.config)
        classifier = TruthfulnessClassifier(self.config)
        classifier.train(train_encodings, y_train, test_encodings, y_test, label_to_id)

        self.assertTrue(os.path.exists(classifier.output_dir))
        self.assertTrue(os.path.exists(os.path.join(classifier.output_dir, 'pytorch_model.bin')))

    def test_prediction(self):
        classifier = TruthfulnessClassifier(self.config)
        with open('tests/examples/statement_data.json', 'r') as f:
            statement_data = json.load(f)
        prediction, explanation = classifier.predict(statement_data)
        self.assertIn(prediction, classifier.label_to_id.keys())
        self.assertIsInstance(explanation, str)


if __name__ == '__main__':
    unittest.main()
