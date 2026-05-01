"""
Script to optimize clustering threshold in methods using Agglomerative Clustering.
"""

import os
import dspy
from clustering_datasets.base import Session
from evaluation.metrics import avg_spk_f1
import tqdm
import numpy as np


def create_examples(sessions: list[Session]) -> list[dspy.Example]:
    """
    Converts a list of Session objects into a list of dspy.Example objects.
    """
    examples = []
    for session in sessions:
        example = dspy.Example(
            transcripts=session.transcripts,
            timestamps=session.timestamps,
            clusters=session.true_labels,
            session_id=session.session_id
        ).with_inputs("transcripts", "timestamps", "session_id")

        examples.append(example)
    return examples


def tune_clustering_threshold(program_class, dataset, model: str, llm_api_endpoint: str, step_size: float = 0.1):
    """
    Tunes the clustering threshold for a given clustering program using the provided dataset.

    NOTES:
    - The program must accept a clustering_threshold parameter. This is rather hacky and should be refactored in the future.
    
    Args:
        program: The clustering program to be evaluated. Must accept a clustering_threshold parameter.
        dataset: The dataset containing training sessions.
        model: The language model to be used.
        llm_api_endpoint: The API endpoint for the language model.
        step_size: The increment step size for threshold values between 0 and 1.

    Returns:
        best_threshold: The threshold value that yielded the highest average speaker F1 score.
        best_score: The highest average speaker F1 score achieved.
        thresholds: List of threshold values evaluated.
        scores: List of corresponding average speaker F1 scores for each threshold.
    """

    train_sessions = dataset.get_train()
    train_examples = create_examples(train_sessions)
    
    thresholds = np.arange(0, 1.0, step_size)
    scores = []

    lm = dspy.LM(
        model=model,
        api_base=llm_api_endpoint,
        temperature=0.0,
        api_key=os.environ.get("API_KEY")
    )
    dspy.configure(lm=lm)

    for thr in tqdm.tqdm(thresholds, desc="Optimizing Clustering Threshold"):
        clustering_program = program_class(clustering_threshold=thr)

        evaluate = dspy.Evaluate(
            devset=train_examples,
            metric=avg_spk_f1,
            display_progress=False,
            provide_traceback=False,
            display_table=False,
            num_threads=7,
        )
        result = evaluate(program=clustering_program)
        scores.append(result.score)

    best_idx = np.argmax(scores)
    best_threshold = thresholds[best_idx]
    best_score = scores[best_idx]

    return best_threshold, best_score, thresholds, scores
