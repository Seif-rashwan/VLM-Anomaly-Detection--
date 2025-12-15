import cv2
import torch
import numpy as np
from PIL import Image
import open_clip
from typing import Iterator, Tuple
# PHASE 5 FIX: Import from central config
from ml_core.config import MODEL_NAME, PRETRAINED_WEIGHTS

# --- Utility to load VLM preprocessor only once ---
_PREPROCESS = None 

def get_vlm_preprocessor():
    """Loads the VLM image preprocessor only once for efficiency."""
    global _PREPROCESS
    if _PREPROCESS is None:
        # Load only the preprocessor, not the full model (saves VRAM)
        # Uses the unified config constants
        _, _, _PREPROCESS = open_clip.create_model_and_transforms(
            MODEL_NAME, 
            pretrained=PRETRAINED_WEIGHTS
        )
    return _PREPROCESS

def extract_sampled_frames(
    video_path: str, 
    sampling_rate_fps: float = 1.0
) -> Iterator[Tuple[torch.Tensor, float]]:
    """
    Generator function that loads a video and yields preprocessed frames one by one.
    """
    preprocess = get_vlm_preprocessor()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file at {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Warning: Could not read video FPS. Assuming 30 FPS.")
        fps = 30.0

    frame_skip_interval = max(1, int(round(fps / sampling_rate_fps)))
    frame_count = 0
    
    print(f"Processing video {video_path} at {sampling_rate_fps} FPS (Source: {fps:.2f} FPS)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 

        if frame_count % frame_skip_interval == 0:
            if is_frame_valid(frame):
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                processed_tensor = preprocess(pil_image)
                current_time_s = frame_count / fps
                
                yield processed_tensor, current_time_s
            
        frame_count += 1
        
    cap.release()

def is_frame_valid(frame: np.ndarray, threshold: int = 10) -> bool:
    """Checks if a frame is valid (not just a black screen)."""
    if frame is None:
        return False
    avg_intensity = np.mean(frame)
    return avg_intensity > threshold