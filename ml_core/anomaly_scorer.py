import numpy as np
from typing import List, Dict
from ml_core.config import DEFAULT_SIGMOID_BIAS, DEFAULT_SIGMOID_TEMP, ANOMALY_THRESHOLD

def compute_anomaly_scores(
    normal_similarities: List[float],
    anomaly_similarities: List[float],
    temperature: float = DEFAULT_SIGMOID_TEMP,
    bias: float = DEFAULT_SIGMOID_BIAS
) -> List[float]:
    """
    Computes anomaly scores using a biased sigmoid function.
    
    Includes safety clipping and configurable sensitivity.
    """
    normal_sims = np.array(normal_similarities)
    anomaly_sims = np.array(anomaly_similarities)
    
    # 1. Calculate raw difference
    raw_scores = anomaly_sims - normal_sims
    
    # 2. SAFETY: Clip extreme values to prevent overflow/underflow
    # CLIP similarity is usually within [-1, 1], so differences are [-2, 2].
    # We clip to [-2.0, 2.0] just to be safe mathematically.
    raw_scores = np.clip(raw_scores, -2.0, 2.0)
    
    # 3. Apply Biased Sigmoid
    # Formula: 1 / (1 + e^(-(x - bias) / temp))
    normalized_scores = 1 / (1 + np.exp(-(raw_scores - bias) / temperature))
    
    return normalized_scores.tolist()

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