import dspy
from clustering_datasets.base import Session

from evaluation.metrics import dev_avg_f1, train_avg_f1


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
            trace_tags={"session_id": session.session_id},
        ).with_inputs("transcripts", "timestamps", "trace_tags")

        examples.append(example)
    return examples


def evaluate(cluster_prog: dspy.Module, dev: list[Session], train: list[Session], lm, display_progress: bool = True, display_table: bool = True):
    """
    Evaluates a clustering program on development and training datasets.
    """
    dspy.configure(lm=lm)

    dev_examples = create_examples(dev)
    train_examples = create_examples(train)

    eval_args = {
        "num_threads": 7,
        "max_errors": 10,
        "display_progress": display_progress,
        "display_table": display_table,
        "return_all_scores": False,
        "provide_traceback": False,
    }

    dev_f1 = None
    train_f1 = None

    if len(dev) > 0:
        dev_eval = dspy.Evaluate(
            devset=dev_examples,
            metric=dev_avg_f1,
            **eval_args
        )

        dev_result = dev_eval(cluster_prog)
        dev_f1 = dev_result.score

    if len(train) > 0:
        train_eval = dspy.Evaluate(
            devset=train_examples,
            metric=train_avg_f1,
            **eval_args
        )
        train_result = train_eval(cluster_prog)
        train_f1 = train_result.score

    return dev_f1, train_f1
