"""
Configuration classes for managing experiment settings.
"""

from dataclasses import dataclass
import yaml


@dataclass
class ClusterConfig:
    method: str
    params: dict

@dataclass
class DatasetConfig:
    name: str
    params: dict

@dataclass
class EvaluationConfig:
    eval_on_train: bool
    eval_on_dev: bool
    num_threads: int
    max_errors: int

@dataclass
class ExperimentConfig:
    name: str
    run_name: str
    cluster: ClusterConfig
    evaluation: EvaluationConfig
    dataset: DatasetConfig

    @classmethod
    def from_dict(cls, input_dict):
        cluster = ClusterConfig(**input_dict['cluster'])
        eval = EvaluationConfig(**input_dict['evaluation'])
        dataset = DatasetConfig(**input_dict['dataset'])

        return cls(
            name=input_dict['experiment']['name'],
            run_name=input_dict['experiment']['run_name'],
            cluster=cluster,
            evaluation=eval,
            dataset=dataset,
        )

    @classmethod
    def from_yaml(cls, exp_path: str):
        """
        Create ExperimentConfig from YAML files.

        Args:
            exp_path (str): Path to the experiment configuration YAML file.
        Returns:
            ExperimentConfig: The constructed ExperimentConfig object.
        """

        with open(exp_path, 'r') as file:
            exp_dict = yaml.safe_load(file)

        # overrides experiment config with general config
        return cls.from_dict(exp_dict)
