import argparse
from evaluation import evalute_with_mlflow
from evaluation.experiment_config import ExperimentConfig
from config import Config
from clustering_datasets.factory import dataset_factory
from cluster.factory import clusterer_factory


if __name__ == "__main__":
    argp = argparse.ArgumentParser(description="Run clustering evaluation.")
    argp.add_argument("path", help='Path to the experiment configuration YAML file.')
    argp.add_argument("--config", type=str, help="Path to the general configuration YAML file.", default="config/config.yaml")
    argp.add_argument("--passive-ratio", type=float, default=0, help="Ratio of speakers to make passive. Default is 0.")
    args = argp.parse_args()

    experiment = ExperimentConfig.from_yaml(args.path)
    config = Config.from_yaml(args.config)


    dataset = dataset_factory(experiment.dataset.name, experiment.dataset.params)
    cluster_prog = clusterer_factory(experiment.cluster.method, experiment.cluster.params)

    evalute_with_mlflow(cluster_prog, dataset, experiment, config, args.passive_ratio)
