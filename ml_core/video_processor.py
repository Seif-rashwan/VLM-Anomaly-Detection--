import cv2
import torch
import numpy as np
from PIL import Image
import open_clip
from typing import Iterator, Tuple, List, Optional
# Import from central config
from ml_core.config import MODEL_NAME, PRETRAINED_WEIGHTS

# --- Utility to load VLM preprocessor only once ---
_PREPROCESS = None 

def get_vlm_preprocessor():
    """Loads the VLM image preprocessor only once for efficiency."""
    global _PREPROCESS
    if _PREPROCESS is None:
        # Load only the preprocessor, not the full model (saves VRAM)
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

import torch.nn.functional as F

def create_patches(frame_tensor: torch.Tensor, grid_size: int = 4) -> Tuple[torch.Tensor, List[Tuple[int, int, int, int]]]:
    """
    Splits a (3, H, W) tensor into a batch of (grid_size^2, 3, 224, 224) tensors.
    """
    c, h, w = frame_tensor.shape
    patch_h, patch_w = h // grid_size, w // grid_size
    
    patches = []
    coords = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            y1 = i * patch_h
            x1 = j * patch_w
            
            # Extract patch
            patch = frame_tensor[:, y1:y1+patch_h, x1:x1+patch_w]
            
            # Resize to CLIP standard
            patch_resized = F.interpolate(
                patch.unsqueeze(0), 
                size=(224, 224), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
            patches.append(patch_resized)
            coords.append((x1, y1, patch_w, patch_h))
            
    return torch.stack(patches), coords

# --- NEW: Resolution-Aware Motion Detection ---
def detect_motion(
    current_frame: np.ndarray, 
    prev_frame: Optional[np.ndarray], 
    threshold: int = 50, 
    min_area: Optional[int] = None
) -> Tuple[bool, List[np.ndarray], np.ndarray]:
    """
    Detects motion between two frames using background subtraction.
    
    Args:
        current_frame: The current video frame (BGR).
        prev_frame: The previous video frame (Grayscale/Blurred).
        threshold: Pixel difference threshold (0-255).
        min_area: Minimum area in pixels to count as motion. 
                  If None, calculated dynamically as 0.2% of frame size.
                  
    Returns:
        motion_detected (bool): True if significant motion found.
        contours (list): List of contours for moving regions.
        processed_frame (ndarray): The processed grayscale frame for the next loop.
    """
    # 1. Preprocess: Grayscale + Blur
    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # 2. Dynamic Resolution Scaling
    # Fix Problem 4: Scale min_area based on resolution
    if min_area is None:
        h, w = gray.shape
        min_area = int(0.001 * (h * w))  # 0.1% of total pixels

    # Initial frame case
    if prev_frame is None:
        return False, [], gray

    # 3. Compute Difference
    frame_delta = cv2.absdiff(prev_frame, gray)
    thresh = cv2.threshold(frame_delta, threshold, 255, cv2.THRESH_BINARY)[1]
    
    # 4. Dilate to fill holes
    thresh = cv2.dilate(thresh, None, iterations=2)

    # 5. Find Contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 6. Filter by Area
    motion_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    motion_detected = len(motion_contours) > 0
    
    return motion_detected, motion_contours, gray