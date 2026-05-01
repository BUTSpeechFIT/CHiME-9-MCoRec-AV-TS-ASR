import os
import dspy
import json
import numpy as np
import tqdm
from clustering_datasets.stm import StmDataset
from evaluation.evaluate import evaluate
from clustering_datasets import Chime9
from cluster.factory import clusterer_factory
import argparse
from config import Config

dspy.disable_logging()

if __name__ == "__main__":
    argp = argparse.ArgumentParser(description="Tune Pairwise Clustering Thresholds")
    argp.add_argument("--train-path", type=str, help="Path to training sessions.", default="data-bin/chime9/train")
    argp.add_argument("--output-dir", type=str, help="Directory to save optimization results.", default="optimization/clustering_threshold")
    argp.add_argument("--config", type=str, help="Path to the general configuration YAML file.", default="config/config.yaml")
    argp.add_argument("--program", type=str, help="Clustering program to use.", default="pairwise")

    args = argp.parse_args()
    dataset = StmDataset()

    config = Config.from_yaml(args.config)

    lm = dspy.LM(
        model=config.llm.model,
        api_base=config.llm.endpoint,
        api_key=os.environ.get("API_KEY"),
        temperature=0.0, # for replicability
    )

    thresholds = np.linspace(0.1, 0.9, 9)
    f1_scores = {}

    train = dataset.get_train()

    # Grid search over overlap and topic thresholds
    for clustering_threshold in tqdm.tqdm( thresholds, desc="Tuning Clustering Threshold"):
        cluster_prog = clusterer_factory(args.program, params={"clustering_threshold": clustering_threshold})

        _, f1 = evaluate(cluster_prog, [], train, lm, display_progress=False, display_table=False)

        f1_scores[clustering_threshold] = f1

    # Find best thresholds
    best_param = max(f1_scores.keys(), key=lambda k: f1_scores[k])
    best_f1 = f1_scores[best_param]
    print(f"Best Clustering Thr: {best_param:.2f}, Best F1 Score: {best_f1:.4f}")

    # Save results
    output_file_path = f"{args.output_dir}/pairwise_thresholds.json"
    os.makedirs(args.output_dir, exist_ok=True)

    with open(output_file_path, "w") as f:
        json.dump({
            "f1_scores": {f"{k}": v for k, v in f1_scores.items()},
            "best_clustering_threshold": best_param,
            "best_f1_score": best_f1,
        }, f, indent=4)

    print(f"Results saved to {output_file_path}")

