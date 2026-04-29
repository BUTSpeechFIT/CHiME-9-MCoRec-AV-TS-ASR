"""
This script showcases how to run custom evaluation and inference using DSPy
"""

import os
import dspy
from cluster.pairwise.hybrid import HybridCluster
from clustering_datasets.factory import dataset_factory
from cluster.factory import clusterer_factory
from evaluation.evaluate import evaluate

def setup_dspy():
    """
    Sets up DSPy with necessary configurations.
    """

    # setup llm
    lm = dspy.LM(
        model="fill_in_your_model",
        api_base="api_url",
        api_key=os.environ.get("API_KEY"),
        temperature=0.0, # for replicability
    )
    dspy.configure(lm=lm)


if __name__ == "__main__":
    # -- setup llm --
    lm = dspy.LM(
        model="fill_in_your_model",
        api_base="api_url",
        api_key=os.environ.get("API_KEY"),
        temperature=0.0, # for replicability
    )
    dspy.configure(lm=lm)

    # -- Load dataset - one of chime9, ami, stm --
    dataset = dataset_factory("chime9", {})
    devset = dataset.get_dev()
    trainset = dataset.get_train()

    # -- Load clusterer --
    # pairwise - predicts same-conversation likelihood score for each pair of segments based on shared topic
    # joint - jointly clusters all segments using LLM
    # pairwise_refined - runs pairwise clustering followed by refinement step using joint clustering
    # hybrid - combines topic scoring with LLM and time based scoring based on segment timestamps
    cluster_prog = clusterer_factory("hybrid")

    # -- Run inference --
    # Here we run inference on the first sample of the dataset
    # All cluster_prog accept same arguments: transcripts and timestamps
    pred_result = cluster_prog(
        transcripts=devset[0].transcripts,
        timestamps=devset[0].timestamps,
    )
    spk_to_clusters = pred_result.clusters

    print("Speaker to cluster assignments:", spk_to_clusters)

    # -- Run evaluation --
    # Here we evaluate on the whole devset and trainset

    # run eval function
    dev_f1, train_f1 = evaluate(
        cluster_prog,
        devset,
        trainset,
        lm,
        display_progress=True,
        display_table=False,
    )

    print(f"Dev F1: {dev_f1}, Train F1: {train_f1}")
