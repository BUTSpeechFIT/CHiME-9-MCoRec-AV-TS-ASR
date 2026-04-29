"""
This script runs inference on speaker clustering using DSPy and a hybrid clustering approach. Uses CHiME-9 evalset as inputs.
"""

import os
import dspy
from cluster.pairwise.hybrid import HybridCluster
from clustering_datasets import Chime9
import argparse
import json
import tqdm
from concurrent.futures import ThreadPoolExecutor

from config import Config

def save_session_pred(cluster_dict, path):
    """
    Saves the prediction result for a session to a specified path.
    """
    os.makedirs(path, exist_ok=True)
    output_file = os.path.join(path, "speaker_to_cluster.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cluster_dict, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    argp = argparse.ArgumentParser(
        description="Run inference on speaker clustering using DSPy and hybrid clustering."
    )
    argp.add_argument( "--sessions-dir", type=str, required=False, default="data-bin/eval-per-word", help="Directory containing session data." )
    argp.add_argument( "--output-dir", type=str, required=False, default="outputs/hybrid_clusterer", help="Directory to save output results." )
    argp.add_argument("--config", type=str, help="Path to the general configuration YAML file.", default="config/config.yaml")
    args = argp.parse_args()

    config = Config.from_yaml(args.config)

    # -- setup llm --
    lm = dspy.LM(
        model=config.llm.model,
        api_base=config.llm.endpoint,
        api_key=os.environ.get("API_KEY"),
        temperature=0.0, # for replicability
    )
    dspy.configure(lm=lm)

    # -- Load dataset --
    dataset = Chime9(
        dev_path=args.sessions_dir,
        labels_dir=""
    )
    data = dataset.get_dev()

    # -- Load clusterer --
    cluster_prog = HybridCluster()

    # -- Run inference --
    def process_session(session):
        pred_result = cluster_prog(
            transcripts=session.transcripts,
            timestamps=session.timestamps,
        )

        save_session_pred(
            cluster_dict=pred_result.clusters,
            path=os.path.join(args.output_dir, session.session_id),
        )

    with ThreadPoolExecutor(max_workers=5) as executor:
        list(
            tqdm.tqdm(
                executor.map(process_session, data),
                total=len(data),
                desc="Processing sessions",
            )
        )

    print(f"Inference completed. Results saved to {args.output_dir}.")
