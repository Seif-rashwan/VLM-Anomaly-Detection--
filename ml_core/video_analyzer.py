import cv2
import torch
import numpy as np
import os
from typing import Dict, List
from ml_core.vlm_test import get_model
from ml_core.anomaly_scorer import compute_anomaly_scores, compute_metadata
from ml_core.video_processor import extract_sampled_frames, create_patches
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
        bounding_boxes = []
        
        print(f"Starting analysis: {os.path.basename(video_path)}")

        for frame_tensor, timestamp in extract_sampled_frames(video_path, sampling_rate_fps):
            
            # Prepare Input: Add batch dimension (3, H, W) -> (1, 3, H, W)
            image_input = frame_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
            # Calculate Similarity & Softmax Score
                logits = (image_features @ text_features.T)
                probs = logits.softmax(dim=-1).cpu().numpy()[0]
                global_score = float(probs[1]) # Anomaly score
            
            # Store raw similarities for smoothing later (legacy support)
            # Note: We rely on the softmax score for the immediate trigger
            normal_similarities.append(float(logits[0, 0].item()))
            anomaly_similarities.append(float(logits[0, 1].item()))
            timestamps.append(timestamp)

            # --- LOCALIZATION LOGIC ---
            best_box = None
            if global_score > ANOMALY_THRESHOLD:
                patch_batch, coords = create_patches(frame_tensor, grid_size=4)
                patch_batch = patch_batch.to(device)
                
                with torch.no_grad():
                    p_features = model.encode_image(patch_batch)
                    p_features /= p_features.norm(dim=-1, keepdim=True)
                    
                    # Batch similarity
                    p_logits = (p_features @ text_features.T)
                    p_probs = p_logits.softmax(dim=-1)[:, 1] # Anomaly column
                    
                    # Find winner
                    best_idx = torch.argmax(p_probs).item()
                    if p_probs[best_idx] > ANOMALY_THRESHOLD:
                        best_box = coords[best_idx] # (x, y, w, h)
            
            bounding_boxes.append(best_box)
            
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
            {"time_s": ts, "score": score, "bbox": bbox}
            for ts, score, bbox in zip(timestamps, anomaly_scores, bounding_boxes)
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