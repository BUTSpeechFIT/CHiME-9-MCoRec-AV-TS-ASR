import itertools
from typing import List, Dict

def pairwise_f1_score(true_labels: List[int], pred_labels: List[int]) -> float:
    """
    Adopted from baseline Chime-9 code.
    Compute the pairwise F1 score for clustering evaluation.
    
    Args:
        true_labels (List[int]): Ground truth cluster labels.
        pred_labels (List[int]): Predicted cluster labels.
    
    Returns:
        float: Pairwise F1 score.
    """
    # Generate all unique unordered pairs of indices
    pairs = list(itertools.combinations(range(len(true_labels)), 2))
    
    # Initialize counts
    tp = fp = fn = 0
    
    for i, j in pairs:
        # True same-cluster?
        true_same = (true_labels[i] == true_labels[j])
        # Predicted same-cluster?
        pred_same = (pred_labels[i] == pred_labels[j])
        
        if pred_same and true_same:
            tp += 1
        elif pred_same and not true_same:
            fp += 1
        elif not pred_same and true_same:
            fn += 1
        # True negatives (not same in both) are not used in F1
    # print(tp, fp, fn)
    # Handle edge cases
    if tp == 0:
        return 0.0
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1

def get_clustering_f1_score(conversation_clusters_label: Dict[str, int], true_clusters_label: Dict[str, int]) -> float:
    """
    Adopted from baseline Chime-9 code.
    Calculate F1 score for clustering results.
    
    Args:
        conversation_clusters_label: Dictionary mapping speaker IDs to cluster IDs
        true_clusters_label: Dictionary mapping speaker IDs to true cluster IDs
        
    Returns:
        F1 score
    """
    spk_list = list(conversation_clusters_label.keys())
    true_clusters = [true_clusters_label[spk] for spk in spk_list]
    conversation_clusters = [conversation_clusters_label[spk] for spk in spk_list]
    return pairwise_f1_score(true_clusters, conversation_clusters)

def avg_spk_f1(example, pred, trace=None, pred_name=None, pred_trace=None):
    """
    DSPY metric function for speaker clustering evaluation.
    """
    avg_f1 = get_clustering_f1_score(
        pred.clusters,
        example.clusters
    )

    # NOTE: might be better to incorporate individual speaker-pairwise scores
    return avg_f1

# Hack to differentiate dev and train metrics in MLflow
def dev_avg_f1(example, pred, trace=None, pred_name=None, pred_trace=None):
    return avg_spk_f1(example, pred, trace, pred_name, pred_trace)

def train_avg_f1(example, pred, trace=None, pred_name=None, pred_trace=None):
    return avg_spk_f1(example, pred, trace, pred_name)
