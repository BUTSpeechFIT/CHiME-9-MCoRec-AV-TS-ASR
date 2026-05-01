"""
Adapted from MCorec baseline implementation.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from typing import List, Dict

MAX_SPEAKERS = 8
MAX_CONVERSATIONS = 4

def cluster_speakers(scores: np.ndarray, 
                    speaker_ids: List[str],
                    threshold: float = 0.7,
                    n_clusters: int|None = None) -> Dict[str, int]:
    """
    Cluster speakers based on their conversation scores.
    
    Args:
        scores: NxN numpy array of conversation scores
        speaker_ids: List of speaker IDs
        threshold: Minimum score to consider speakers in same conversation
        n_clusters: Number of clusters (if None, will be determined automatically)
        
    Returns:
        Dictionary mapping cluster IDs to lists of speaker IDs
    """
    if n_clusters is not None and n_clusters > MAX_CONVERSATIONS:
        raise ValueError(f"Maximum number of conversations is {MAX_CONVERSATIONS}")
    
    # Convert scores to distance matrix (1 - score)
    distances = 1 - scores
    
    # Perform hierarchical clustering
    if n_clusters is None:
        # Use threshold to determine number of clusters
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1-threshold,
            metric='precomputed',
            linkage='complete'
        )
    else:
        clustering = AgglomerativeClustering(
            n_clusters=min(n_clusters, MAX_CONVERSATIONS),
            metric='precomputed',
            linkage='complete'
        )
    
    cluster_labels = clustering.fit_predict(distances)
    
    # Group speakers by cluster
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(speaker_ids[i])
    
    # Speaker to cluster mapping
    spk_to_cluster = {spk_id: label.item() for spk_id, label in zip(speaker_ids, cluster_labels)}
    return spk_to_cluster
