# API Contract Specification: VLM Anomaly Detection

## Purpose

This document defines the strict output structure for the core backend analysis function, ensuring the ML Core Team and UI/UX Team can integrate seamlessly in Phase 3.

## Backend Function Signature (ML Core Team)
The main callable function must adhere to:
```python
def analyze_video(video_path: str, prompt_normal: str, prompt_anomaly: str) -> dict:
    # Implementation details...
    return analysis_results # Must match the JSON schema below
## Required Output Schema (JSON Format)

The function must return a Python dictionary that serializes exactly to the following JSON structure. **This structure includes the time-series data required for the UI's line chart.** 

### 1. Root Structure

| Key | Type | Description |
| :--- | :--- | :--- |
| `status` | `str` | Either `"Success"` or `"Error"`. |
| `data` | `list[dict]` | **CRITICAL:** The core time-series data for the line chart visualization. |
| `metadata` | `dict` | Summary statistics and timestamp for the anomaly highlight. |

### 2. Time-Series Data (`data` Array Structure)

This array contains the data points for the UI's line chart, aggregated by segment time (e.g., 1 point per second), which aligns with best practices for time-series analysis.[2]

| Key (inside each dictionary) | Type | Constraint | Description |
| :--- | :--- | :--- | :--- |
| `time_s` | `float` | $time\_s \ge 0.0$ | The timestamp of the segment **in seconds**. |
| `score` | `float` | $0.0 \le score \le 1.0$ | The final computed anomaly score for that segment, **scaled from 0.0 to 1.0**. |

**Example of the required array structure:**
```json
"data": [
    {"time_s": 0.0, "score": 0.15},
    {"time_s": 1.0, "score": 0.95}, // High anomaly segment
    {"time_s": 2.0, "score": 0.30},
  ...
]
