import unittest
import os
import yaml
import optuna
from optuna.trial import TrialState
from truthfulness_classifier.optuna_tuning import objective, load_config

class TestOptunaTuning(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config_path = 'config.yaml'
        cls.data_path = 'tests/examples/data.csv'
        cls.config = load_config(cls.config_path)

    def test_optuna_tuning(self):
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, self.data_path, self.config), n_trials=2)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        self.assertGreater(len(study.trials), 0)
        self.assertGreaterEqual(len(pruned_trials), 0)
        self.assertGreater(len(complete_trials), 0)

        best_trial = study.best_trial
        self.assertIsNotNone(best_trial)
        self.assertLessEqual(best_trial.value, 1.0)  # Assuming the loss is normalized

if __name__ == '__main__':
    unittest.main()
