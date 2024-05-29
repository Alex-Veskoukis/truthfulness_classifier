import unittest
import os
from truthfulness_classifier.optuna_tuning import main as optuna_main


class TestOptunaTuning(unittest.TestCase):

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

    def test_optuna_tuning(self):
        import sys
        sys.argv = ['optuna_tuning', '--data', self.data_path]
        try:
            optuna_main()
        except SystemExit:
            pass  # Prevent the script from exiting the test runner


if __name__ == '__main__':
    unittest.main()
