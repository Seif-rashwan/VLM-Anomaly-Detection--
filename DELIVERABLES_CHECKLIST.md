## File Structure Summary

```
VLM-Anomaly-Detection--/
├── ml_core/
│   ├── vlm_test.py          # Phase 1: ML Prototype
│   ├── video_analyzer.py    # Phase 2: Video Processing Pipeline
│   └── anomaly_scorer.py    # Phase 2: Anomaly Scoring Algorithm
├── ui/
│   ├── app.py               # Phase 1 & 3: Image Analysis UI
│   └── app_video.py         # Phase 2 & 3: Video Analysis UI
├── data/
│   ├── test_image.jpg       # Test file for image analysis
│   └── test_video.mp4       # Test file for video analysis
├── requirements.txt         # All dependencies
└── DELIVERABLES_CHECKLIST.md # This file
```

---

## Verification Commands

### Test Image Analysis (Phase 1):
```bash
streamlit run ui/app.py
```

### Test Video Analysis (Phase 2 & 3):
```bash
streamlit run ui/app_video.py
```

### Test ML Backend Directly:
```bash
python ml_core/vlm_test.py
```


