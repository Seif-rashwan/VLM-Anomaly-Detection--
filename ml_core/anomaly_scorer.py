import numpy as np
from typing import List, Dict
from ml_core.config import DEFAULT_SIGMOID_BIAS, DEFAULT_SIGMOID_TEMP, ANOMALY_THRESHOLD

def compute_anomaly_scores(
    normal_similarities: List[float],
    anomaly_similarities: List[float],
    temperature: float = None, # Deprecated
    bias: float = None         # Deprecated
) -> List[float]:
    """
    Returns the anomaly probabilities directly.
    
    Previous versions used a sigmoid on logits. 
    Now that we use Softmax probabilities [0, 1], we just return them.
    """
    # Simply return the anomaly probability (presumed to be in 0.0-1.0 range)
    return list(anomaly_similarities)

def compute_metadata(
    anomaly_scores: List[float],
    timestamps: List[float],
    total_frames: int,
    total_seconds: float
) -> Dict:
    """
    Computes metadata from anomaly scores.
    """
    scores_array = np.array(anomaly_scores)
    timestamps_array = np.array(timestamps)
    
    # SAFETY: Early exit for empty data
    if len(scores_array) == 0:
        return {
            "total_frames": 0,
            "total_seconds": 0,
            "max_anomaly_score": 0.0,
            "max_anomaly_time": 0.0,
            "average_score": 0.0,
            "min_score": 0.0,
            "max_score": 0.0,
            "std_score": 0.0,
            "anomaly_threshold": ANOMALY_THRESHOLD,
            "is_anomaly_detected": False
        }

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
        "std_score": float(np.std(scores_array)),
        # Standardize the threshold decision here
        "anomaly_threshold": ANOMALY_THRESHOLD,
        "is_anomaly_detected": max_score >= ANOMALY_THRESHOLD
    }