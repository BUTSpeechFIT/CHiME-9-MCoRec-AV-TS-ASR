import dspy
import mlflow
from typing import Dict, List, Tuple


class BaseCluster(dspy.Module):
    def __init__(self):
        super().__init__()

    def _forward(self, transcripts, timestamps, **kwargs):
        raise NotImplementedError()

    def forward(
        self,
        transcripts: Dict[str, list[str]],
        timestamps: Dict[str, List[Tuple[float, float]]],
        **kwargs,
    ) -> dspy.Prediction:
        # Log trace tags if provided
        if kwargs.get("trace_tags", None) and mlflow.active_run() is not None:
                mlflow.update_current_trace(tags=kwargs["trace_tags"])

        return self._forward(transcripts, timestamps, **kwargs)
