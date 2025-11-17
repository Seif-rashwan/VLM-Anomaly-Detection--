# Testing Guide for VLM Anomaly Detection System

This guide helps you test all deliverables across Phase 1, Phase 2, and Phase 3.

## Prerequisites

1. Install all dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have test files:
   - `data/test_image.jpg` - For image analysis
   - `data/test_video.mp4` - For video analysis

## Phase 1 Testing: ML Prototype

### Test 1: Command-Line ML Prototype
```bash
python ml_core/vlm_test.py
```

**Expected Output:**
- Model loads successfully
- Similarity scores printed for normal and anomaly prompts
- Example: "Similarity to 'Dog walking peacefully': 0.2753"

### Test 2: Image Analysis UI
```bash
streamlit run ui/app.py
```

**Test Steps:**
1. Upload `data/test_image.jpg` (or any image)
2. Enter normal prompt: "Dog walking peacefully"
3. Enter anomaly prompt: "A heavily damaged car after a crash."
4. Click " Start Analysis"
5. Verify similarity scores are displayed
6. Verify classification (NORMAL or ANOMALY) is shown

**Expected Results:**
-  Image displays correctly
-  Similarity scores appear (two metrics)
-  Classification result shown
-  Detailed results expander works

## Phase 2 Testing: Video Processing

### Test 3: Video Processing Pipeline
Create a test script `test_video_pipeline.py`:
```python
from ml_core.video_analyzer import analyze_video

results = analyze_video(
    video_path="data/test_video.mp4",
    prompt_normal="A person is walking normally.",
    prompt_anomaly="A person is falling down.",
    sampling_rate_fps=1.0
)

print(f"Status: {results['status']}")
print(f"Data points: {len(results['data'])}")
print(f"Metadata: {results['metadata']}")
```

**Expected Results:**
-  Video loads successfully
-  Frames are processed
-  Returns list of anomaly scores over time
-  Metadata includes total_frames, total_seconds, max_anomaly_score, etc.

### Test 4: Anomaly Scoring Algorithm
```python
from ml_core.anomaly_scorer import compute_anomaly_scores

normal_sims = [0.8, 0.7, 0.6, 0.3, 0.2]
anomaly_sims = [0.2, 0.3, 0.4, 0.7, 0.8]

scores = compute_anomaly_scores(normal_sims, anomaly_sims)
print(f"Anomaly scores: {scores}")
# Should output normalized scores in [0, 1] range
```

**Expected Results:**
-  Scores normalized to [0, 1] range
-  Higher anomaly similarity = higher score
-  Lower normal similarity = higher score

## Phase 3 Testing: Integrated System

### Test 5: Full Video Analysis UI
```bash
streamlit run ui/app_video.py
```

**Test Steps:**
1. Upload `data/test_video.mp4` (or any video)
2. Enter normal prompt: "A person is walking normally."
3. Enter anomaly prompt: "A person is falling down."
4. Set sampling rate: 1.0 fps (default)
5. Click " Start Anomaly Analysis"

**Expected Results:**
-  Video preview displays
-  Progress bar shows during processing
-  Analysis completes successfully
-  **Prominent message**: " Anomaly detected at MM:SS" (if score > 0.5)
-  Summary metrics displayed (4 columns)
-  **Line chart** showing anomaly score over time
-  Raw data table available in expander
-  Detailed results expander works

### Test 6: Results Visualization Verification

**Checklist:**
- [ ] Line chart displays correctly
- [ ] X-axis shows time in seconds
- [ ] Y-axis shows score [0, 1]
- [ ] Anomaly timestamp message is prominent (top of results)
- [ ] Time formatted as MM:SS
- [ ] Chart is interactive and responsive

### Test 7: Integration Testing

**Test with 2-3 Sample Videos:**

1. **Normal Video Test:**
   - Upload video of normal activity
   - Prompts: Normal="Person walking", Anomaly="Person falling"
   - Expected: Low anomaly scores, no prominent anomaly message

2. **Anomaly Video Test:**
   - Upload video with anomaly event
   - Prompts: Normal="Person walking", Anomaly="Person falling"
   - Expected: High anomaly score at event time, prominent message shown

3. **Edge Cases:**
   - Very short video (< 5 seconds)
   - Very long video (> 2 minutes)
   - Different video formats (MP4, MOV, AVI)

## Quick Verification Script

Run this to verify all core functions work:

```python
# test_all_components.py
import sys
sys.path.append('.')

print("Testing ML Components...")

# Test 1: Image Analysis
from ml_core.vlm_test import analyze_image_vlm
from PIL import Image

try:
    img = Image.open("data/test_image.jpg")
    result = analyze_image_vlm(img, "normal", "anomaly")
    print(f" Image analysis: {result}")
except Exception as e:
    print(f" Image analysis failed: {e}")

# Test 2: Video Analysis
from ml_core.video_analyzer import analyze_video

try:
    result = analyze_video(
        "data/test_video.mp4",
        "normal activity",
        "anomaly event",
        sampling_rate_fps=1.0
    )
    print(f" Video analysis status: {result['status']}")
    print(f" Data points: {len(result.get('data', []))}")
except Exception as e:
    print(f" Video analysis failed: {e}")

# Test 3: Anomaly Scoring
from ml_core.anomaly_scorer import compute_anomaly_scores

try:
    scores = compute_anomaly_scores([0.8, 0.7], [0.2, 0.3])
    print(f" Anomaly scoring: {scores}")
except Exception as e:
    print(f" Anomaly scoring failed: {e}")

print("\nAll core components tested!")
```

## Common Issues & Solutions

### Issue: ModuleNotFoundError
**Solution**: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: Video won't load
**Solution**: Check video format is supported (MP4, MOV, AVI, MKV, WebM)

### Issue: Model loading is slow
**Solution**: First run downloads model weights. Subsequent runs use cached model.

### Issue: Streamlit warnings about PyTorch
**Solution**: These are harmless warnings. The lazy loading in the code minimizes them.

## Performance Benchmarks

Expected processing times (approximate):
- Image analysis: 1-3 seconds (first run: 10-30s for model download)
- Video analysis: ~1-2 seconds per second of video (at 1 fps sampling rate)
  - 10-second video: ~10-20 seconds
  - 60-second video: ~60-120 seconds

## Success Criteria

All tests should pass with:
-  No errors in console
-  All UI elements display correctly
-  Results are accurate and meaningful
-  Performance is acceptable (< 2 min for 1-minute video)

---

**Status**: Ready for comprehensive testing! 

