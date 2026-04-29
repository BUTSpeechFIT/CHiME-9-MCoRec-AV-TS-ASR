"""
DSPY module to separate speakers into clusters of different conversations based on their transcripts.

@author Hai Phong Nguyen
@date 16.1.2026
"""

from collections import defaultdict
import dspy
import numpy as np
from cluster.base import BaseCluster
from cluster.common import calculate_conversation_scores
from typing import Dict, List, Optional, Tuple

class JointClusterSignature(dspy.Signature):
    """Seperate speakers into clusters of different conversations based on their transcripts. Same conversations tend to have mutual topic and subjects."""

    transcripts: Dict[str, list[str]] = dspy.InputField(desc="A dictionary mapping speaker IDs to their transcripts. Each key is a speaker ID.")

    # overlap_ratios: Dict[str, Dict[str, float]] = dspy.InputField(desc="Time overlap ratio (0-1) between pairs of speakers, indicating percentage of segment time overlaps between them. 0 = no overlap, 1 = complete overlap. High overlap ratio suggests speakers are less likely to be in the same conversation, altough it is not definitive.")

    clusters: dict[str, int] = dspy.OutputField(desc="Same keys as transcripts, mapping each speaker ID to a conversation cluster.")


class JointCluster(BaseCluster):
    def __init__(self):
        self.cluster = dspy.ChainOfThought(JointClusterSignature)


    def _forward(
        self,
        transcripts: Dict[str, list[str]],
        timestamps: Dict[str, List[Tuple[float, float]]],
        **kwargs,
    ) -> dspy.Prediction:
        overlap_scores_mat = calculate_conversation_scores(timestamps, likelihood=False)
        speaker_ids = list(transcripts.keys())
        overlap_scores = defaultdict(dict)

        for i,spk_i in enumerate(speaker_ids):
            for j, spk_j in enumerate(speaker_ids):
                if spk_i == spk_j:
                    continue

                overlap_scores[spk_i][spk_j] = overlap_scores_mat[i,j]

        clusters = self.cluster(
            transcripts=transcripts, 
            # overlap_ratios=overlap_scores_mat
        ).clusters

        return dspy.Prediction(
            clusters=clusters
        )
