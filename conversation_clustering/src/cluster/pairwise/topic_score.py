import dspy
from typing import Tuple, Dict

class TopicConversationScore(dspy.Signature):
    """Predict score of how likely two speakers are in the same conversation. Speakers in the same conversation tend to have mutual topic, and low overlaps between them."""

    transcripts: Dict[str, list[str]] = dspy.InputField(desc="A dictionary mapping speaker IDs to their transcripts. Each key is a speaker ID.")

    topic_similarity: float = dspy.OutputField(desc="Score between 0-1 indicating topic similarity between the two speakers. 0 = completely different topics, 1 = same topic.")

    same_conversation_likelihood: float = dspy.OutputField(desc="Score between 0-1 indicating likelihood of being in the same conversation. 0 = different conversations, 1 = same conversation.")


class PredictTopicConversationScore(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict_pairwise_score = dspy.ChainOfThought(TopicConversationScore)

    def forward(
        self,
        speaker_a_segments: Tuple[str, list],
        speaker_b_segments: Tuple[str, list],
        **kwargs,
    ) -> dspy.Prediction:

        score_pred = self.predict_pairwise_score(
            transcripts={
                speaker_a_segments[0]: speaker_a_segments[1],
                speaker_b_segments[0]: speaker_b_segments[1]
            },
        )

        return dspy.Prediction(
            score=score_pred.topic_similarity
        )
