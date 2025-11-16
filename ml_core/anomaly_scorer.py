import numpy as np
from typing import List, Dict, Tuple

def compute_anomaly_scores(
    normal_similarities: List[float],
    anomaly_similarities: List[float]
) -> List[float]:
    """
    Computes anomaly scores from normal and anomaly similarity scores.
    
    The anomaly score is calculated as:
    - Higher anomaly similarity = higher anomaly score
    - Lower normal similarity = higher anomaly score
    - Score is normalized to [0, 1] range
    
    Args:
        normal_similarities: List of similarity scores to normal prompt
        anomaly_similarities: List of similarity scores to anomaly prompt
    
    Returns:
        List of anomaly scores in [0, 1] range
    """
    normal_sims = np.array(normal_similarities)
    anomaly_sims = np.array(anomaly_similarities)
    
    # Calculate raw anomaly score
    # When anomaly similarity is high and normal similarity is low, score should be high
    # We use the difference and normalize
    raw_scores = anomaly_sims - normal_sims
    
    # Normalize to [0, 1] range
    # CLIP similarities are typically in [-1, 1] range, so differences are in [-2, 2]
    # We shift and scale to [0, 1]
    min_score = raw_scores.min()
    max_score = raw_scores.max()
    
    if max_score - min_score > 0:
        normalized_scores = (raw_scores - min_score) / (max_score - min_score)
    else:
        # If all scores are the same, return 0.5 (neutral)
        normalized_scores = np.full_like(raw_scores, 0.5)
    
    return normalized_scores.tolist()

def compute_metadata(
    anomaly_scores: List[float],
    timestamps: List[float],
    total_frames: int,
    total_seconds: float
) -> Dict:
    """
    Computes metadata from anomaly scores.
    
    Args:
        anomaly_scores: List of anomaly scores
        timestamps: List of timestamps corresponding to scores
        total_frames: Total number of frames processed
        total_seconds: Total duration of video in seconds
    
    Returns:
        Dictionary with metadata including max score, peak time, average, etc.
    """
    scores_array = np.array(anomaly_scores)
    timestamps_array = np.array(timestamps)
    
    max_score_idx = np.argmax(scores_array)
    max_score = float(scores_array[max_score_idx])
    max_anomaly_time = float(timestamps_array[max_score_idx])
    average_score = float(np.mean(scores_array))
    
    return {
        "total_frames": total_frames,
        "total_seconds": total_seconds,
        "max_anomaly_score": max_score,
        "max_anomaly_time": max_anomaly_time,
        "average_score": average_score,
        "min_score": float(np.min(scores_array)),
        "max_score": float(np.max(scores_array)),
        "std_score": float(np.std(scores_array))
    }
