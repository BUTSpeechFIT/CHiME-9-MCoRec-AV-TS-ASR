import dspy
from typing import Dict, List, Optional, Tuple
from .base import BaseCluster
from util import cluster_speakers
from cluster.common import calculate_conversation_scores
import mlflow


class BaselineCluster(BaseCluster):
    def __init__(self, clustering_threshold: float = 0.7):
        self.clustering_threshold = clustering_threshold

    # @mlflow.trace(name="BaselineCluster Forward Pass")
    def forward(
        self,
        transcripts: Dict[str, list[str]],
        timestamps: Optional[Dict[str, List[Tuple[float,float]]]] = None,
        session_id: str = "",
        **kwargs,
    ) -> dspy.Prediction:

        if timestamps is None:
            raise ValueError("Timestamps are required for BaselineCluster.")


        scores = calculate_conversation_scores(timestamps)

        mlflow.start_span("BaselineCluster Forward Pass")
        span = mlflow.get_current_active_span()
        mlflow.update_current_trace(tags={"session_id": session_id})

        if span:
            span.set_inputs({
                "scores": scores.tolist(),
            })

        clusters = cluster_speakers(
            scores,
            list(timestamps.keys()),
            threshold=self.clustering_threshold
        )

        return dspy.Prediction(
            clusters=clusters
        )
