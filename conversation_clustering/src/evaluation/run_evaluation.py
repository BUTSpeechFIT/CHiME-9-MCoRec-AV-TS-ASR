import os
import dspy
import mlflow
import mlflow.dspy as mlflow_dspy
from clustering_datasets.base import AbstractDataset
from .experiment_config import ExperimentConfig
from config import Config
from .evaluate import evaluate

def evalute_with_mlflow(cluster_prog: dspy.Module, dataset: AbstractDataset, experiment: ExperimentConfig, config: Config, passive_per_speaker: int = 0):
    """
    Runs evaluation of a clustering program on a given dataset and logs results to MLflow.
    """
    dev = dataset.get_dev_with_passive(passive_per_speaker) if experiment.evaluation.eval_on_dev else []
    train = dataset.get_train_with_passive(passive_per_speaker) if experiment.evaluation.eval_on_train else []

    # setup mlflow
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow_dspy.autolog(log_compiles=False, log_evals=True, log_traces_from_compile=False)
    mlflow.set_experiment(experiment_name=experiment.name)

    # setup dspy
    lm = dspy.LM(
        model=config.llm.model,
        api_base=config.llm.endpoint,
        api_key=os.environ.get("API_KEY"),
        temperature=0.0, # for replicability
    )

    # fallback to default if no run name is provided
    run_name = experiment.run_name or f"{experiment.cluster.method}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("llm_model", config.llm.model)
        mlflow.log_param("method", cluster_prog.__class__.__name__)
        mlflow.log_param("dev_size", len(dev))
        mlflow.log_param("train_size", len(train))

        # log clustering parameters
        for param, value in experiment.cluster.params.items():
            mlflow.log_param(param, value)

        # log dataset parameters
        for param, value in experiment.dataset.params.items():
            mlflow.log_param(param, value)

        dev_f1, train_f1 = evaluate(cluster_prog, dev, train, lm)

        if dev_f1 is not None:
            mlflow.log_metric("dev_f1", dev_f1)
        if train_f1 is not None:
            mlflow.log_metric("train_f1", train_f1)
        if dev_f1 is not None and train_f1 is not None:
            avg_f1 = (dev_f1 * len(dev) + train_f1 * len(train)) / (len(dev) + len(train))
            mlflow.log_metric("avg_f1", avg_f1)
