import dataclasses

import yaml

from model.config import ModelConfig


def load_config(path: str) -> ModelConfig:
    """Load a YAML config file into a ModelConfig.

    Unknown keys are ignored so configs can carry extra metadata without
    breaking construction; derived fields (e.g. d_k) are never read from YAML.
    """
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    valid = {f.name for f in dataclasses.fields(ModelConfig)}
    filtered = {k: v for k, v in data.items() if k in valid}
    return ModelConfig(**filtered)
