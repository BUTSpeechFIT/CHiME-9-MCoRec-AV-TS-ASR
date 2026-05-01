import os
import dspy
import json
import numpy as np
import tqdm
from evaluation.evaluate import evaluate
from clustering_datasets import Chime9
from cluster import HybridCluster
import argparse
from config import Config

dspy.disable_logging()


if __name__ == "__main__":
    argp = argparse.ArgumentParser(description="Tune Hybrid Clustering Thresholds")
    argp.add_argument("--passive-per-session", type=int, default=0, help="How many speakers to passivate per session during evaluation. Default is 0.")
    argp.add_argument("--train-path", type=str, help="Path to training sessions.", default="data-bin/chime9/train")
    argp.add_argument("--output-dir", type=str, help="Directory to save optimization results.", default="optimization/clustering_threshold")
    argp.add_argument("--config", type=str, help="Path to the general configuration YAML file.", default="config/config.yaml")
    argp.add_argument()

    args = argp.parse_args()
    cluster_prog = HybridCluster()
    dataset = Chime9(train_path=args.train_path)

    config = Config.from_yaml(args.config)

    lm = dspy.LM(
        model=config.llm.model,
        api_base=config.llm.endpoint,
        api_key=os.environ.get("API_KEY"),
        temperature=0.0, # for replicability
    )

    thresholds = np.linspace(0.1, 0.9, 9)
    f1_scores = {}

    train = dataset.get_train_with_passive(args.passive_per_session)

    # Grid search over overlap and topic thresholds
    for overlap_thr, topic_thr in tqdm.tqdm( [(o, t) for o in thresholds for t in thresholds], desc="Tuning Overlap and Topic Thresholds"):
            cluster_prog.set_overlap_threshold(overlap_thr)
            cluster_prog.set_topic_threshold(topic_thr)

            _, f1 = evaluate(cluster_prog, [], train, lm, display_progress=False, display_table=False)

            f1_scores[(overlap_thr, topic_thr)] = f1

    # Find best thresholds
    best_params = max(f1_scores.keys(), key=lambda k: f1_scores[k])
    best_f1 = f1_scores[best_params]
    print(f"Best Overlap Thr: {best_params[0]:.2f}, Best Topic Thr: {best_params[1]:.2f}, Best F1 Score: {best_f1:.4f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/hybrid_thresholds_{args.passive_per_session}.json", "w") as f:
        json.dump({
            "f1_scores": {f"{k[0]}_{k[1]}": v for k, v in f1_scores.items()},
            "best_overlap_threshold": best_params[0],
            "best_topic_threshold": best_params[1],
            "best_f1_score": best_f1,
        }, f, indent=4)

    print(f"Results saved to {args.output_dir}/hybrid_thresholds_{args.passive_per_session}.json")
