import cv2
import torch
import numpy as np
import os
from typing import Dict, List
from ml_core.vlm_test import get_model
from ml_core.anomaly_scorer import compute_anomaly_scores, compute_metadata
from ml_core.video_processor import extract_sampled_frames, create_patches
try:
    from ml_core.config import (
        ANOMALY_THRESHOLD, 
        DEFAULT_SIGMOID_BIAS, 
        DEFAULT_SIGMOID_TEMP
    )
except ImportError:
    ANOMALY_THRESHOLD = 0.6
    DEFAULT_SIGMOID_BIAS = 0.05
    DEFAULT_SIGMOID_TEMP = 0.1

def analyze_video(
    video_path: str,
    prompt_normal: str,
    prompt_anomaly: str,
    sampling_rate_fps: float = 1.0
) -> Dict:
    """
    Analyzes a video using VLM to detect anomalies frame-by-frame.
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
        
        # 1. Get Video Metadata
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
             return {"status": "Error", "error_message": "Invalid video file", "data": [], "metadata": {}}
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_seconds = total_frames / fps if fps > 0 else 0
        cap.release()

        # 2. Load Model
        model, tokenizer, preprocess, device = get_model()
        model.eval()
        
        prompts = [prompt_normal, prompt_anomaly]
        text_input = tokenizer(prompts).to(device)
        
        with torch.no_grad():
            text_features = model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # 3. Process Frames
        normal_similarities = []
        anomaly_similarities = []
        timestamps = []
        bounding_boxes = []
        
        print(f"Starting analysis: {os.path.basename(video_path)}")

        for frame_tensor, timestamp in extract_sampled_frames(video_path, sampling_rate_fps):
            
            image_input = frame_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # 1. Calculate Scaled Logits
                # Multiply by 100.0 to scale cosine similarity for Softmax
                logits = (image_features @ text_features.T) * 100.0
                
                # 2. Compute Softmax Probabilities (Standardized Scoring)
                probs = logits.softmax(dim=-1).cpu().numpy()[0]
                prob_normal = float(probs[0])
                prob_anomaly = float(probs[1])
                
                global_score = prob_anomaly
                
            normal_similarities.append(prob_normal)
            anomaly_similarities.append(prob_anomaly)
            timestamps.append(timestamp)

            # --- LOCALIZATION LOGIC ---
            best_box = None
            
            if global_score > ANOMALY_THRESHOLD:
                patch_batch, coords = create_patches(frame_tensor, grid_size=4)
                patch_batch = patch_batch.to(device)
                
                with torch.no_grad():
                    p_features = model.encode_image(patch_batch)
                    p_features /= p_features.norm(dim=-1, keepdim=True)
                    
                    # Scale patch logits too!
                    p_logits = (p_features @ text_features.T) * 100.0
                    p_probs = p_logits.softmax(dim=-1)[:, 1] 
                    
                    best_idx = torch.argmax(p_probs).item()
                    
                    if p_probs[best_idx] > ANOMALY_THRESHOLD:
                        best_box = coords[best_idx]
            
            bounding_boxes.append(best_box)
            
            if len(timestamps) % 50 == 0:
                print(f"Processed {len(timestamps)} frames...")

        if not timestamps:
             return {
                "status": "Success",
                "data": [],
                "metadata": compute_metadata([], [], total_frames, total_seconds)
            }

        # 4. Compute Final Scores
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