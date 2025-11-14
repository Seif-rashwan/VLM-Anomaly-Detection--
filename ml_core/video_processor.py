import cv2
import torch
import numpy as np
import open_clip

# Global variable to load the preprocessor once (efficiency)
# In Phase 3, this will be handled by the ml_api_handler
_PREPROCESS = None 

def get_vlm_preprocessor(model_name: str = "ViT-B-16", pretrained: str = "laion2b_s34b_b88k"):
    """Loads the VLM image preprocessor only once."""
    global _PREPROCESS
    if _PREPROCESS is None:
        # We only need the preprocessor transform here
        _, _, _PREPROCESS = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained
        )
    return _PREPROCESS

def extract_sampled_frames(video_path: str, sampling_rate_fps: int = 1) -> list:
    """
    Loads a video, extracts frames at a low sampling rate (temporal segmentation),
    and preprocesses them for VLM input.

    Returns a list of VLM-ready PyTorch Tensors.
    """
    preprocess = get_vlm_preprocessor()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file at {video_path}")
        return

    # 1. Determine Sampling Interval (The Efficiency Hack)
    fps = cap.get(cv2.CAP_PROP_FPS) # Source video FPS (e.g., 30.0)
    if fps == 0:
        print("Warning: Could not read video FPS. Assuming 30 FPS.")
        fps = 30.0

    # frame_skip_interval: E.g., if fps=30 and we want 1 FPS, skip 30 frames.
    frame_skip_interval = max(1, int(round(fps / sampling_rate_fps))) 
    
    frame_count = 0
    sampled_tensors =
    timestamps =

    while cap.isOpened():
        ret, frame = cap.read() # ret=success flag, frame=image (BGR numpy array)
        
        if not ret:
            break 
        
        if frame_count % frame_skip_interval == 0:
            # 2. Frame Preprocessing and Conversion
            
            # Convert OpenCV BGR (Blue-Green-Red) format to PIL RGB (Required by VLM preprocessor)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert NumPy array to PIL Image and apply VLM transformations (scaling, normalization)
            pil_image = Image.fromarray(rgb_frame)
            processed_tensor = preprocess(pil_image)
            
            # Append the VLM-ready tensor
            sampled_tensors.append(processed_tensor)
            
            # Calculate and store the timestamp (in seconds) for this segment
            current_time_s = frame_count / fps
            timestamps.append(current_time_s)
            
        frame_count += 1
        
    cap.release()
    print(f"Source FPS: {fps:.2f}. Target Sampling Rate: {sampling_rate_fps} FPS.")
    print(f"Total frames processed: {frame_count}. Sampled tensors created: {len(sampled_tensors)}")
    
    return sampled_tensors, timestamps
