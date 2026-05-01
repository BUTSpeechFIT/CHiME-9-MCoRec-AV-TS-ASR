import sys
import os

from numpy import disp

from clustering_datasets.base import Session
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from cluster.pairwise.process_pair import PredictTopicConversationScore
import json
import dspy
from mlflow import dspy as mlflow_dspy
import mlflow
import numpy as np
from evaluation.metrics import avg_spk_f1
from cluster.common import calculate_conversation_scores
from evaluation.metrics import get_clustering_f1_score


def create_examples(sessions: list[Session]) -> list[dspy.Example]:
    """
    Converts a list of Session objects into a list of dspy.Example objects.
    """
    examples = []
    for session in sessions:
        spk_ids = list(session.transcripts.keys())
        overlap_ratios = calculate_conversation_scores(session.timestamps)

        for i, spk_a_id in enumerate(spk_ids):
            for j, spk_b_id in enumerate(spk_ids):
                if i >= j:
                    continue

                true_label = int(session.true_labels[spk_a_id] == session.true_labels[spk_b_id])

                example = dspy.Example(
                    speaker_a_segments=session.transcripts[spk_a_id],
                    speaker_b_segments=session.transcripts[spk_b_id],
                    overlap_ratio=overlap_ratios[i,j],
                    true_label=true_label,
                ).with_inputs("speaker_a_segments", "speaker_b_segments", "overlap_ratio")

                examples.append(example)
    return examples


def pair_metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    try:
        predicted_score = pred['score']
    except KeyError:
        feedback_text = "Prediction does not contain 'score' key."
        return dspy.Prediction(
            score = 0.0,
            feedback = feedback_text
        )

    true_label = example['true_label']

    feedback_text = ""
    if true_label:
        if predicted_score >= 0.5:
            if predicted_score >= 0.9:
                feedback_text = "You correctly identified that the speakers are in the same conversation."
            else:
                feedback_text = "You correctly identified that the speakers are in the same conversation, but with low confidence. Is there any additional information that could help increase your confidence?"
        else:
            feedback_text = "The speakers are in the same conversation. However, you predicted they are not. Please review the segments again and consider any cues that indicate they belong to the same conversation."
    else:
        if predicted_score < 0.5:
            if predicted_score <= 0.1:
                feedback_text = "You correctly identified that the speakers are in different conversations."
            else:
                feedback_text = "You correctly identified that the speakers are in different conversations, but with low confidence. Is there any additional information that could help increase your confidence?"
        else:
            feedback_text = "The speakers are in different conversations. However, you predicted they are in the same conversation. Please review the segments again and consider any cues that indicate they belong to different conversations."

    score = 1.0 if (predicted_score >= 0.5) == (true_label == 1) else 0.0
    return dspy.Prediction(
        score = score,
        feedback = feedback_text
    )


def optimize_pairwise_prompts(dataset, model: str, llm_api_endpoint: str):
    train_sessions = dataset.get_train()
    dev_sessions = dataset.get_dev()
    train_examples = create_examples(train_sessions)
    dev_examples = create_examples(dev_sessions)[:10] # GEPA requires smaller dev set

    
    lm = dspy.LM(
        model=model,
        api_base=llm_api_endpoint,
        temperature=0.0,
        api_key=os.environ.get("API_KEY")
    )
    dspy.configure(lm=lm)

    reflection_lm = dspy.LM(
        model="openai/deepseek-r1",
        api_base=llm_api_endpoint,
        api_key=os.environ.get("API_KEY")
    )

    prog = PredictTopicConversationScore()

    mlflow.set_tracking_uri("sqlite:///mlflow/cluster_experiments_new.sqlite")
    mlflow_dspy.autolog(log_compiles=True, log_evals=False, log_traces_from_compile=True)
    mlflow.set_experiment(experiment_name="Pairwise Optimization")

    with mlflow.start_run(run_name=f"Pairwise_Optimization_{dataset.__class__.__name__}"):
        mlflow.log_param("llm_model", model)
        mlflow.log_param("dataset", dataset.__class__.__name__)
        mlflow.log_param("train_size", len(train_examples))
        mlflow.log_param("dev_size", len(dev_examples))

        optimize = dspy.GEPA(
            metric=pair_metric,
            auto="light",
            track_stats=True,
            reflection_minibatch_size=5,
            reflection_lm=reflection_lm
        )

        optimized_program = optimize.compile(prog,trainset=train_examples, valset=dev_examples)

        os.makedirs("optimized_pairwise_program", exist_ok=True)
        optimized_program.save("optimized_pairwise_program", save_program=True)
