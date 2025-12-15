import cv2
import torch
import numpy as np
import os
from typing import Dict, List
from ml_core.vlm_test import get_model
from ml_core.anomaly_scorer import compute_anomaly_scores, compute_metadata
from ml_core.video_processor import extract_sampled_frames
# If you implemented the config step, import it. Otherwise, use defaults.
try:
    from ml_core.config import ANOMALY_THRESHOLD
except ImportError:
    ANOMALY_THRESHOLD = 0.7

def analyze_video(
    video_path: str,
    prompt_normal: str,
    prompt_anomaly: str,
    sampling_rate_fps: float = 1.0
) -> Dict:
    """
    Analyzes a video using VLM to detect anomalies frame-by-frame.
    
    PHASE 3 FIX: Removed duplicate video loop. Now uses the
    'extract_sampled_frames' generator from video_processor.py.
    """
    try:
        # Validate video file
        if not os.path.exists(video_path):
            return {
                "status": "Error",
                "error_message": f"Video file not found: {video_path}",
                "data": [],
                "metadata": {}
            }
        
        # 1. Get Video Metadata (Quick Open/Close)
        # We perform a quick check to get total duration for the report.
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
             return {"status": "Error", "error_message": "Invalid video file", "data": [], "metadata": {}}
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_seconds = total_frames / fps if fps > 0 else 0
        cap.release()

        # 2. Load Model & Prepare Text Embeddings
        model, tokenizer, preprocess, device = get_model()
        model.eval() # Ensure evaluation mode
        
        prompts = [prompt_normal, prompt_anomaly]
        text_input = tokenizer(prompts).to(device)
        
        with torch.no_grad():
            text_features = model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # 3. Process Frames (DE-DUPLICATED LOOP)
        # We now iterate over the generator. Logic is defined in ONE place (video_processor).
        normal_similarities = []
        anomaly_similarities = []
        timestamps = []
        
        print(f"Starting analysis: {os.path.basename(video_path)}")

        for frame_tensor, timestamp in extract_sampled_frames(video_path, sampling_rate_fps):
            
            # Prepare Input: Add batch dimension (3, H, W) -> (1, 3, H, W)
            image_input = frame_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Calculate Similarity
                similarity = (image_features @ text_features.T).squeeze(0)
                similarity_scores = similarity.cpu().numpy()
            
            # Store scores
            normal_similarities.append(float(similarity_scores[0]))
            anomaly_similarities.append(float(similarity_scores[1]))
            timestamps.append(timestamp)
            
            # Optional: Progress logging
            if len(timestamps) % 50 == 0:
                print(f"Processed {len(timestamps)} frames...")

        # Edge Case: No frames processed (e.g., all black frames)
        if not timestamps:
             return {
                "status": "Success",
                "data": [],
                "metadata": compute_metadata([], [], total_frames, total_seconds)
            }

        # 4. Compute Final Scores & Metadata
        anomaly_scores = compute_anomaly_scores(normal_similarities, anomaly_similarities)
        
        data = [
            {"time_s": ts, "score": score}
            for ts, score in zip(timestamps, anomaly_scores)
        ]
        
        metadata = compute_metadata(
            anomaly_scores=anomaly_scores,
            timestamps=timestamps,
            total_frames=total_frames,
            total_seconds=total_seconds
        )
        
        return {
            "status": "Success",
            "data": data,
            "metadata": metadata
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "Error",
            "error_message": str(e),
            "data": [],
            "metadata": {}
        }