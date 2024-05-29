import unittest
import os
from truthfulness_classifier.model_training import TruthfulnessClassifier
from truthfulness_classifier.data_preprocessing import preprocess_data


class TestModelTraining(unittest.TestCase):

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
        self.model_dir = './test_model_dir'
        os.makedirs(self.model_dir, exist_ok=True)

    def tearDown(self):
        os.remove(self.data_path)
        if os.path.exists(self.model_dir):
            for file in os.listdir(self.model_dir):
                file_path = os.path.join(self.model_dir, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            os.rmdir(self.model_dir)

    def test_model_training(self):
        train_encodings, test_encodings, y_train, y_test, tokenizer, label_to_id = preprocess_data(self.data_path)
        classifier = TruthfulnessClassifier(num_labels=len(label_to_id), output_dir=self.model_dir)
        classifier.train(train_encodings, y_train, test_encodings, y_test, label_to_id)
        self.assertTrue(os.path.exists(self.model_dir))


if __name__ == '__main__':
    unittest.main()
