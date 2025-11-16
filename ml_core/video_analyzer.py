import cv2
import torch
import numpy as np
from PIL import Image
import os
from typing import Dict, List
from ml_core.vlm_test import get_model
from ml_core.anomaly_scorer import compute_anomaly_scores, compute_metadata

def analyze_video(
    video_path: str,
    prompt_normal: str,
    prompt_anomaly: str,
    sampling_rate_fps: float = 1.0
) -> Dict:
    """
    Analyzes a video using VLM to detect anomalies frame-by-frame.
    
    Args:
        video_path: Path to the video file
        prompt_normal: Text description of normal condition
        prompt_anomaly: Text description of anomaly condition
        sampling_rate_fps: Frames per second to sample (default: 1.0)
    
    Returns:
        Dictionary with status, data (time-series), and metadata
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
        
        # Load VLM model (cached)
        model, tokenizer, preprocess, device = get_model()
        
        # Tokenize prompts once
        prompts = [prompt_normal, prompt_anomaly]
        text_input = tokenizer(prompts).to(device)
        
        # Calculate text embeddings once
        with torch.no_grad():
            text_features = model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                "status": "Error",
                "error_message": f"Cannot open video file: {video_path}",
                "data": [],
                "metadata": {}
            }
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30.0  # Default assumption
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_seconds = total_frames / fps if fps > 0 else 0
        
        # Calculate frame skip interval
        frame_skip_interval = max(1, int(round(fps / sampling_rate_fps)))
        
        # Process frames
        frame_count = 0
        normal_similarities = []
        anomaly_similarities = []
        timestamps = []
        
        print(f"Processing video: {os.path.basename(video_path)}")
        print(f"FPS: {fps:.2f}, Total frames: {total_frames}, Sampling rate: {sampling_rate_fps} fps")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames based on sampling rate
            if frame_count % frame_skip_interval == 0:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                
                # Preprocess image
                image_input = preprocess(pil_image).unsqueeze(0).to(device)
                
                # Calculate image embedding
                with torch.no_grad():
                    image_features = model.encode_image(image_input)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    
                    # Calculate similarity scores
                    similarity = (image_features @ text_features.T).squeeze(0)
                    similarity_scores = similarity.cpu().numpy()
                
                # Store results
                normal_similarities.append(float(similarity_scores[0]))
                anomaly_similarities.append(float(similarity_scores[1]))
                timestamps.append(frame_count / fps)
            
            frame_count += 1
        
        cap.release()
        
        # Compute anomaly scores
        anomaly_scores = compute_anomaly_scores(normal_similarities, anomaly_similarities)
        
        # Create time-series data
        data = [
            {"time_s": ts, "score": score}
            for ts, score in zip(timestamps, anomaly_scores)
        ]
        
        # Compute metadata
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
        return {
            "status": "Error",
            "error_message": str(e),
            "data": [],
            "metadata": {}
        }

