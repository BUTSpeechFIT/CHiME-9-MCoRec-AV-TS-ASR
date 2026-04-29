from .chime9 import Chime9
from .ami import AMI
from .stm import StmDataset
from .base import AbstractDataset

DATASET_REGISTRY = {
    "chime9": Chime9,
    "ami": AMI,
    "stm": StmDataset,
}

def dataset_factory(dataset_name, params=None) -> AbstractDataset:
    if params is None:
        params = {}

    try:
        cls = DATASET_REGISTRY[dataset_name]
    except KeyError:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return cls(**params)
