import unittest
import os
from truthfulness_classifier.data_preprocessing import preprocess_data, preprocess_single_statement
import pandas as pd


class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        self.data_path = 'tests/examples/data.csv'
        data = {
            'statement': ["This is a test statement."],
            'subjects': ["test"],
            'speaker_name': ["Test Speaker"],
            'speaker_job': ["Tester"],
            'speaker_state': ["Test State"],
            'speaker_affiliation': ["Test Affiliation"],
            'statement_context': ["Test Context"],
            'Label': ["true"]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.data_path, index=False)

    def tearDown(self):
        os.remove(self.data_path)

    def test_preprocess_data(self):
        train_encodings, test_encodings, y_train, y_test, tokenizer, label_to_id = preprocess_data(self.data_path)
        self.assertIsNotNone(train_encodings)
        self.assertIsNotNone(test_encodings)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_test)
        self.assertIsNotNone(tokenizer)
        self.assertIsNotNone(label_to_id)

    def test_preprocess_single_statement(self):
        statement_data = {
            "statement": "This is a test statement.",
            "subjects": "test",
            "speaker_name": "Test Speaker",
            "speaker_job": "Tester",
            "statement_context": "Test Context"
        }
        tokenizer = 'huawei-noah/TinyBERT_General_4L_312D'
        encoding = preprocess_single_statement(statement_data, tokenizer)
        self.assertIsNotNone(encoding)


if __name__ == '__main__':
    unittest.main()
