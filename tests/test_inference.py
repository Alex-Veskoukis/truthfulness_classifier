import unittest
import json
from truthfulness_classifier.inference import predict_truthfulness, load_model


class TestInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_dir = 'tests/model_20240528_214408'
        cls.config_path = 'config.yaml'
        cls.statement_data_path = 'tests/examples/statement_data.json'

    def test_load_model(self):
        model, tokenizer, label_to_id = load_model(self.model_dir)
        self.assertIsNotNone(model)
        self.assertIsNotNone(tokenizer)
        self.assertIsNotNone(label_to_id)

    def test_predict_truthfulness(self):
        with open(self.statement_data_path, 'r') as f:
            statement_data = json.load(f)
        prediction, explanation = predict_truthfulness(statement_data,
                                                       model_dir=self.model_dir,
                                                       config_path=self.config_path)
        self.assertIn(prediction, ['true', 'false', 'half-true', 'barely-true'])
        self.assertIsInstance(explanation, str)

if __name__ == '__main__':
    unittest.main()
