import dspy
import numpy as np
from typing import List, Dict, Tuple

from sklearn.cluster import AgglomerativeClustering
from cluster.base import BaseCluster
from cluster.common import calculate_conversation_scores, calculate_overlap_duration
from .topic_score import PredictTopicConversationScore
from collections import defaultdict


class DetectTopic(dspy.Signature):
    """
    Determine whether it is possible to infer general topic of what the speaker is talking about based on the transcript.
    """

    transcript: list[str] = dspy.InputField(desc="Transcript of one speaker")
    contains_topic: bool = dspy.OutputField(desc="Whether it is possible to detect topic or subject from the transcript.")


def group_by_distance(distance_matrix: np.ndarray, items: list, threshold: float) -> Dict[int, list]:
    if len(items) == 0:
        return {}
    elif len(items) == 1:
        return {0: [items[0]]}

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1-threshold,
        metric='precomputed',
        linkage='complete'
    )
    cluster_labels = clustering.fit_predict(distance_matrix)
    clusters = defaultdict(list)
    for item, label in zip(items, cluster_labels):
        clusters[int(label)].append(item)

    return dict(clusters)

def merge_timestamps(timestamps, ids):
    merged = []
    for spk_id in ids:
        merged.extend(timestamps[spk_id])
    merged.sort(key=lambda x: x[0])
    return merged


def calculate_overlap_distance(timestamps_a, timestamps_b):
        overlap, non_overlap = calculate_overlap_duration(
            timestamps_a,
            timestamps_b
        )
        
        # Calculate overlap ratio
        if overlap + non_overlap > 0:
            # Normalize overlap by total duration to get overlap ratio
            total_duration = overlap + non_overlap
            overlap_ratio = overlap / total_duration
        else:
            overlap_ratio = 1.0

        return overlap_ratio


class HybridCluster(BaseCluster):
    def __init__(self, overlap_cl_thr: float = 0.6, topic_cl_thr: float = 0.2, **kwargs):
        self.process_pair = PredictTopicConversationScore()
        self.detect_topic = dspy.Predict(DetectTopic)
        self.overlap_threshold = overlap_cl_thr
        self.topic_threshold = topic_cl_thr

    def set_topic_threshold(self, threshold: float):
        self.topic_threshold = threshold

    def set_overlap_threshold(self, threshold: float):
        self.overlap_threshold = threshold


    def _forward(
        self,
        transcripts: Dict[str, list[str]],
        timestamps: Dict[str, List[Tuple[float, float]]],
        **kwargs,
    ) -> dspy.Prediction:

        with_detectable_topic = []

        for spk_id, transcript in transcripts.items():
            topic_result = self.detect_topic(transcript=transcript)
            if topic_result.contains_topic:
                with_detectable_topic.append(spk_id)


        # Calculate topic-based scores for those with detectable topics
        topic_distances = np.zeros((len(with_detectable_topic), len(with_detectable_topic)))
        for i in range(len(with_detectable_topic)):
            for j in range(i + 1, len(with_detectable_topic)):
                topic_distances[i, j] = 1 - self.process_pair(
                    speaker_a_segments=(with_detectable_topic[i], transcripts[with_detectable_topic[i]]),
                    speaker_b_segments=(with_detectable_topic[j], transcripts[with_detectable_topic[j]]),
                ).score
                topic_distances[j, i] = topic_distances[i, j]

        # Cluster based on topic distances
        topic_clusters = group_by_distance(
            distance_matrix = topic_distances,
            items = with_detectable_topic,
            threshold = self.topic_threshold
        )

        overlap_spk_distances_mat = calculate_conversation_scores(timestamps, likelihood=False)

        # enable indexing by speaker ID
        overlap_spk_distances = {spk_id: {
                other_spk_id: overlap_spk_distances_mat[i][j]
                for j, other_spk_id in enumerate(transcripts.keys())
            } for i, spk_id in enumerate(transcripts.keys())
        }

        without_detectable_topic = [
            spk_id
            for spk_id in transcripts.keys()
            if spk_id not in with_detectable_topic
        ]
        nodes = list(topic_clusters.values()) + list(without_detectable_topic)
        outsider_offset = len(topic_clusters)

        node_distances =  np.zeros((len(nodes), len(nodes)))

        # set distances between topic clusters to maximum
        for i in range(len(topic_clusters)):
            for j in range(len(topic_clusters)):
                if i != j:
                    node_distances[i][j] = 1

        # fill in distances between outsiders and topic clusters
        for oidx, outsider_id in enumerate(without_detectable_topic):
            for cidx, member_ids in enumerate(topic_clusters.values()):
                distances = [
                    overlap_spk_distances[outsider_id][member_id]
                    for member_id in member_ids
                ]
                # # combine timestamps of the members
                # cluster_timestamps = merge_timestamps(timestamps, member_ids)
                #
                # # calculate overlap ratio between outsider and cluster
                # outs_to_cl_dist = calculate_overlap_distance(
                #     timestamps[outsider_id],
                #     cluster_timestamps
                # )

                outs_to_cl_dist = np.mean(distances)

                node_distances[outsider_offset + oidx][cidx] = outs_to_cl_dist
                node_distances[cidx][outsider_offset + oidx] = outs_to_cl_dist

        # fill in distances between outsiders
        for i,spk_id_a in enumerate(without_detectable_topic):
            for j, spk_id_b in enumerate(without_detectable_topic):
                    node_distances[outsider_offset + i][outsider_offset + j] = overlap_spk_distances[spk_id_a][spk_id_b]

        # Cluster nodes based on distances
        final_clusters_raw = group_by_distance(
            distance_matrix = node_distances,
            items = nodes,
            threshold = self.overlap_threshold
        )

        final_clusters = {}
        for cluster_id, member_nodes in final_clusters_raw.items():
            for node in member_nodes:
                if isinstance(node, list):
                    for spk_id in node:
                        final_clusters[spk_id] = cluster_id
                else:
                    final_clusters[node] = cluster_id

        return dspy.Prediction(
            clusters=final_clusters,
            with_detectable_topic=with_detectable_topic,
            without_detectable_topic=list(without_detectable_topic),
            node_distances=node_distances.tolist(),
            overlap_spk_distances=overlap_spk_distances_mat.tolist(),
            topic_clusters=topic_clusters,
            nodes=nodes,
        )
