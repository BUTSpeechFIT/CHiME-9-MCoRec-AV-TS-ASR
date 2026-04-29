from dataclasses import dataclass
import yaml

@dataclass
class LLMConfig:
    model: str
    endpoint: str

@dataclass
class MlflowConfig:
    tracking_uri: str

@dataclass
class Config:
    llm: LLMConfig
    mlflow: MlflowConfig

    @classmethod
    def from_dict(cls, input_dict):
        mlflow = MlflowConfig(**input_dict['mlflow'])
        llm = LLMConfig(**input_dict['llm'])

        return cls(
            llm=llm,
            mlflow=mlflow,
        )

    @classmethod
    def from_yaml(cls, conf_path: str):
        with open(conf_path, 'r') as file:
            exp_dict = yaml.safe_load(file)

        # overrides experiment config with general config
        return cls.from_dict(exp_dict)
