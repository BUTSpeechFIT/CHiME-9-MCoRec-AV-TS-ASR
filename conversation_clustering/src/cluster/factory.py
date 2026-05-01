from .joint import JointCluster
import dspy
from .pairwise import PairwiseCluster, HybridCluster, PairwiseWithRefinement
from .baseline import BaselineCluster

CLUSTER_REGISTRY = {
    "joint": JointCluster,
    "pairwise": PairwiseCluster,
    "pairwise_with_refinement": PairwiseWithRefinement,
    "hybrid": HybridCluster,
    "baseline": BaselineCluster,
}

def clusterer_factory(method, params=None) -> dspy.Module:
    if params is None:
        params = {}

    try:
        cls = CLUSTER_REGISTRY[method]
    except KeyError:
        raise ValueError(f"Unknown cluster method: {method}")
    return cls(**params)
