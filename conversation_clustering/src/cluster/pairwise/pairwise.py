import dspy
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from typing import List, Dict, Tuple
from util import cluster_speakers
from cluster.base import BaseCluster
from .topic_score import PredictTopicConversationScore 

class RefineClustersSignature(dspy.Signature):
    """
    Based on initial clusters that were created using pairwise topic similarity scoring, your are now given the whole context of all speakers. Check whether any speakers should be re-assigned to different clusters based on their and other speakers' transcripts.
    """

    transcripts: Dict[str, list[str]] = dspy.InputField(desc="A dictionary mapping speaker IDs to their transcripts. Each key is a speaker ID.")

    topic_similarity_clusters: dict[str, int] = dspy.InputField(desc="Initial clusters based on topic similarity. Same keys as transcripts, mapping each speaker ID to a conversation cluster.")

    clusters: dict[str, int] = dspy.OutputField(desc="Refined clusters mapping each speaker ID to a conversation cluster. Can be the same as topic_similarity_clusters if no refinement is needed.")

class PairwiseWithRefinement(BaseCluster):
    """
    Cluster speakers into conversations using a two-phase approach:
    1. Pairwise Scoring: Compute pairwise scores based on topic similarity and cluster using Agglomerative Clustering.
    2. Refinement: Jointly refince clusters by supplying overlap ratios and transcripts.
    """
    def __init__(self, **kwargs):
        self.pairwise_cluster = PairwiseCluster(**kwargs)
        self.refine_clusters = dspy.ChainOfThought(RefineClustersSignature)

    def forward(
        self,
        transcripts: Dict[str, list[str]],
        timestamps: Dict[str, List[Tuple[float, float]]],
        **kwargs,
    ) -> dspy.Prediction:

        initial_clusters = self.pairwise_cluster(
            transcripts=transcripts,
            timestamps=timestamps,
            **kwargs
        ).clusters

        refined_clusters = self.refine_clusters(
            transcripts=transcripts,
            topic_similarity_clusters=initial_clusters,
        ).clusters

        return dspy.Prediction(
            clusters=refined_clusters
        )

def cluster_connected_components(matrix: np.ndarray, speaker_ids: List[str]) -> Dict[str, int]:
    graph = csr_matrix(matrix)
    _, labels = connected_components(graph, directed=False, connection='strong')

    spk_to_cluster = {spk_id: label.item() for spk_id, label in zip(speaker_ids, labels)}
    return spk_to_cluster


class PairwiseCluster(BaseCluster):
    def __init__(self, clustering_threshold: float = 0.5, **kwargs):
        self.process_pair = PredictTopicConversationScore(**kwargs)
        self.clustering_threshold = clustering_threshold

    def _forward(
        self,
        transcripts: Dict[str, list[str]],
        timestamps: Dict[str, List[Tuple[float, float]]],
        **kwargs,
    ) -> dspy.Prediction:

        scores = np.zeros((len(transcripts), len(transcripts)))
        spk_ids = list(transcripts.keys())

        for i in range(len(transcripts)):
            for j in range(i + 1, len(transcripts)):
                scores[i, j] = self.process_pair(
                    speaker_a_segments=(spk_ids[i], transcripts[spk_ids[i]]),
                    speaker_b_segments=(spk_ids[j], transcripts[spk_ids[j]]),
                )

        clusters = cluster_speakers(
            scores,
            spk_ids,
            threshold=self.clustering_threshold
        )

        return dspy.Prediction(
            clusters=clusters
        )

