from setuptools import setup, find_packages

setup(
    name='truthfulness_classifier',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'transformers',
        'torch',
        'joblib',
        'optuna',
        'imblearn',
        'nltk',
        'numpy',
        'spacy'

    ],
    entry_points={
        'console_scripts': [
            'train-truthfulness=truthfulness_classifier.model_training:main',
            'predict-truthfulness=truthfulness_classifier.inference:main',
            'tune-truthfulness=truthfulness_classifier.optuna_tuning:main',
        ],
    },
)
