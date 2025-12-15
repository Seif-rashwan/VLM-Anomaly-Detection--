# ml_core/config.py

# --- Model Configuration ---
MODEL_NAME = "ViT-B-16"
PRETRAINED_WEIGHTS = "laion2b_s34b_b88k"

# --- Anomaly Detection Parameters ---
# Threshold: Scores >= 0.7 are flagged as "Anomaly" in the UI
ANOMALY_THRESHOLD = 0.7 

# Sigmoid Parameters (Safety & Sensitivity)
# bias=0.15 shifts the center so "ambiguous" frames get low scores (~0.18)
# temperature=0.1 makes the transition sharp once evidence is found
DEFAULT_SIGMOID_BIAS = 0.05
DEFAULT_SIGMOID_TEMP = 0.1