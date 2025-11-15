import cv2
import torch
import numpy as np
from PIL import Image
import open_clip

# --- Utility to load VLM preprocessor only once ---
_PREPROCESS = None 
_MODEL_NAME = "ViT-B-16"
_PRETRAINED_WEIGHTS = "laion2b_s34b_b88k"

def get_vlm_preprocessor():
    """Loads the VLM image preprocessor only once for efficiency."""
    global _PREPROCESS
    if _PREPROCESS is None:
        _, _, _PREPROCESS = open_clip.create_model_and_transforms(
            _MODEL_NAME, 
            pretrained=_PRETRAINED_WEIGHTS
        )
    return _PREPROCESS

def extract_sampled_frames(video_path: str, sampling_rate_fps: int = 1) -> tuple:
    """
    Loads a video, extracts frames at a low sampling rate (temporal segmentation),
    and preprocesses them into VLM-ready PyTorch Tensors.

    Returns: (list of PyTorch Tensors, list of timestamps in seconds)
    """
    preprocess = get_vlm_preprocessor()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file at {video_path}")
        return [], []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Warning: Could not read video FPS. Assuming 30 FPS.")
        fps = 30.0

    frame_skip_interval = max(1, int(round(fps / sampling_rate_fps)))

    frame_count = 0
    sampled_tensors = []
    timestamps = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 

        if frame_count % frame_skip_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            processed_tensor = preprocess(pil_image)

            sampled_tensors.append(processed_tensor)

            current_time_s = frame_count / fps
            timestamps.append(current_time_s)
            
        frame_count += 1
        
    cap.release()
    print(f"Source FPS: {fps:.2f}. Sampled frames: {len(sampled_tensors)}")
    print(f"Processing load reduced by factor {frame_skip_interval}.")

    return sampled_tensors, timestamps
