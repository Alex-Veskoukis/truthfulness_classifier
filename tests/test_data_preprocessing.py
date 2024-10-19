import unittest
import pandas as pd
from transformers import BertTokenizer
from truthfulness_classifier.data_preprocessing import preprocess_data, preprocess_single_statement
import json

from truthfulness_classifier.inference import load_config


class TestDataPreprocessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.data_path = 'tests/examples/data.csv'
        cls.tokenizer_name = cls.config['model']['name']
        cls.tokenizer = BertTokenizer.from_pretrained(cls.tokenizer_name)

    def test_preprocess_data(self):
        train_encodings, test_encodings, y_train, y_test, tokenizer, label_to_id = preprocess_data(self.data_path,
                                                                                                   self.config)

        total_samples = len(pd.read_csv(self.data_path))
        expected_test_samples = int(total_samples * self.config['data']['test_size'])
        actual_test_samples = len(test_encodings['input_ids'])

        # Allow for rounding differences
        self.assertTrue(abs(actual_test_samples - expected_test_samples) <= 1)
        self.assertEqual(len(train_encodings['input_ids']), total_samples - actual_test_samples)
        self.assertEqual(tokenizer.name_or_path, self.tokenizer_name)
        self.assertGreater(len(label_to_id), 0)

    def test_preprocess_single_statement(self):
        with open('tests/examples/statement_data.json', 'r') as f:
            statement_data = json.load(f)

        encoding = preprocess_single_statement(statement_data, self.tokenizer, self.config['model']['max_length'])

        self.assertIn('input_ids', encoding)
        self.assertIn('attention_mask', encoding)
        self.assertEqual(encoding['input_ids'].shape[1], self.config['model']['max_length'])


if __name__ == '__main__':
    unittest.main()
