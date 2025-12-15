import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ml_core.anomaly_scorer import compute_anomaly_scores

def test_scorer():
    # Case 1: Consistent "Normal" (Anomaly=0, Normal=1) -> Raw = -1
    normal = [1.0] * 10
    anomaly = [0.0] * 10
    scores = compute_anomaly_scores(normal, anomaly)
    print(f"Consistent Normal: {scores[0]}") 
    # Current behavior: 0.5 (Expected: 0.0)

    # Case 2: Consistent "Anomaly" (Anomaly=1, Normal=0) -> Raw = 1
    normal = [0.0] * 10
    anomaly = [1.0] * 10
    scores = compute_anomaly_scores(normal, anomaly)
    print(f"Consistent Anomaly: {scores[0]}") 
    # Current behavior: 0.5 (Expected: 1.0)
    
    # Case 3: Mixed
    normal = [1.0, 0.5]
    anomaly = [0.0, 0.5]
    scores = compute_anomaly_scores(normal, anomaly)
    print(f"Mixed: {scores}")

if __name__ == "__main__":
    test_scorer()
