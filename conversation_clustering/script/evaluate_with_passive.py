"""
Evaluate the effect of varying passive speaker ratios on the performance of the HybridCluster algorithm. Generates and saves plots of F1 scores against passive speaker ratios for both development and training datasets
"""

import argparse
import os
from clustering_datasets import Chime9
from evaluation.evaluate import evaluate
from evaluation.experiment_config import *
from cluster import HybridCluster
import matplotlib.pyplot as plt
import dspy
import numpy as np
import tqdm
from config import Config

dspy.disable_logging()

if __name__ == "__main__":
    argp = argparse.ArgumentParser(description="Evaluate Pairwise Cluster Overlap with varying passive speaker ratios.")
    argp.add_argument("--overlap-threshold", type=float, default=0.6, help="Threshold for overlap clustering for PairwiseClusterOverlap.")
    argp.add_argument("--topic-threshold", type=float, default=0.2, help="Threshold for topic clustering for PairwiseClusterOverlap.")
    argp.add_argument("--output-dir", type=str, help="Directory to save evaluation results.", default="outputs/passive_trend")
    argp.add_argument("--config-path", type=str, help="Path to the configuration file.", default="config/config.yaml")
    argp.add_argument("--train-path", type=str, help="Path to the training sessions.", default="data-bin/chime9/train")
    argp.add_argument("--dev-path", type=str, help="Path to the dev sessions.", default="data-bin/chime9/dev")
    args = argp.parse_args()

    dataset = Chime9(train_path=args.train_path, dev_path=args.dev_path)
    config = Config.from_yaml(args.config_path)

    lm = dspy.LM(
        model = config.llm.model,
        api_base=config.llm.endpoint,
        api_key=os.environ.get("API_KEY"),
        temperature=0.0, # for replicability
    )

    cluster_prog = HybridCluster(overlap_cl_thr=args.overlap_threshold, topic_cl_thr=args.topic_threshold)
    print(f"Evaluating HybridCluster with Overlap Thr: {args.overlap_threshold}, Topic Thr: {args.topic_threshold}")

    dev_f1_scores = []
    train_f1_scores = []

    passive_ratios = np.linspace(0.0, 1, 10)
    for ratio in tqdm.tqdm(passive_ratios, desc="Evaluating Passive Ratios"):
        dev = dataset.get_dev_with_passive(ratio)
        train = dataset.get_train_with_passive(ratio)

        dev_f1, train_f1 = evaluate(
            cluster_prog,
            dev,
            train,
            lm,
            display_progress=False,
            display_table=False,
        )

        dev_f1_scores.append(dev_f1)
        train_f1_scores.append(train_f1)

    # plot and save results
    os.makedirs(args.output_dir, exist_ok=True)
    plt.title("Effect of Passive Speaker Ratio on F1 Score")
    plt.xlabel("Passive Speaker Ratio")
    plt.ylabel("F1 Score")
    plt.plot(passive_ratios, dev_f1_scores, marker='o')
    plt.grid()
    plt.savefig(f"{args.output_dir}/hybrid_cluster_trend_dev.png")
    plt.clf()
    plt.title("Effect of Passive Speaker Ratio on F1 Score")
    plt.xlabel("Passive Speaker Ratio")
    plt.ylabel("F1 Score")
    plt.plot(passive_ratios, train_f1_scores, marker='o', color='orange')
    plt.grid()
    plt.savefig(f"{args.output_dir}/hybrid_cluster_trend_train.png")

    print(f"Plots saved to {args.output_dir}/hybrid_cluster_trend_*.png")

