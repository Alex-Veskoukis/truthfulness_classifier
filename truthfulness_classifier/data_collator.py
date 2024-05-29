import torch
from transformers import DefaultDataCollator

class CustomDataCollator(DefaultDataCollator):
    def __call__(self, features):
        if isinstance(features[0], dict):
            batch = {key: torch.stack([f[key] for f in features]) for key in features[0]}
        else:
            # Handling case where features are tuples
            batch = {
                "input_ids": torch.stack([f[0] for f in features]),
                "attention_mask": torch.stack([f[1] for f in features]),
                "labels": torch.stack([f[2] for f in features])
            }
        return batch



