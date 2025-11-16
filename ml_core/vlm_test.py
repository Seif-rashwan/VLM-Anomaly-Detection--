import torch
import open_clip
from PIL import Image
import os
import numpy as np
from typing import Union, Dict, Tuple

# --- Configuration ---
# Use a smaller, efficient model for testing (e.g., ViT-B/16)
MODEL_NAME = "ViT-B-16"
PRETRAINED_WEIGHTS = "laion2b_s34b_b88k"
IMAGE_PATH = os.path.join("..", "data", "test_image.jpg")

# Global model cache to avoid reloading
_model_cache = None
_tokenizer_cache = None
_preprocess_cache = None
_device_cache = None

def get_model():
    """Load and cache the model to avoid reloading on each call."""
    global _model_cache, _tokenizer_cache, _preprocess_cache, _device_cache
    
    if _model_cache is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, _, preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, 
            pretrained=PRETRAINED_WEIGHTS, 
            device=device
        )
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        
        _model_cache = model
        _tokenizer_cache = tokenizer
        _preprocess_cache = preprocess
        _device_cache = device
    
    return _model_cache, _tokenizer_cache, _preprocess_cache, _device_cache

# --- Core Logic ---

def analyze_image_vlm(
    image: Union[str, Image.Image], 
    normal_prompt: str, 
    anomaly_prompt: str
) -> Dict[str, float]:
    """
    Analyzes an image using VLM (CLIP) to calculate similarity scores for normal and anomaly prompts.
    
    Args:
        image: Either a file path (str) or a PIL Image object
        normal_prompt: Text description of what should be "normal"
        anomaly_prompt: Text description of what should be "anomaly"
    
    Returns:
        Dictionary with 'normal' and 'anomaly' similarity scores
    """
    # Load model (cached)
    model, tokenizer, preprocess, device = get_model()
    
    # Load image if path provided, otherwise use PIL Image directly
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image not found at {image}")
        pil_image = Image.open(image).convert("RGB")
    else:
        pil_image = image.convert("RGB") if image.mode != "RGB" else image
    
    # Process image
    image_input = preprocess(pil_image).unsqueeze(0).to(device)
    
    # Tokenize prompts
    prompts = [normal_prompt, anomaly_prompt]
    text_input = tokenizer(prompts).to(device)
    
    # Calculate embeddings
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
    
    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Calculate similarity scores
    similarity = (image_features @ text_features.T).squeeze(0)
    similarity_scores = similarity.cpu().numpy()
    
    # Return results
    return {
        "normal": float(similarity_scores[0]),
        "anomaly": float(similarity_scores[1])
    }

def run_vlm_prototype(image_path: str):
    """
    Loads CLIP, processes an image and two prompts, and calculates similarity scores.
    (Legacy function for command-line testing)
    """
    if not os.path.exists(image_path):
        print(f"Error: Test image not found at {image_path}. Please place a file there.")
        return

    # Define Test Prompts
    prompts = {
        "normal": "Dog walking peacfully",
        "anomaly": "A heavily damaged car after a crash." 
    }

    # Run analysis
    results = analyze_image_vlm(image_path, prompts["normal"], prompts["anomaly"])
    
    # --- Output and Verification ---
    print("\n--- VLM Prototype Results ---")
    print(f"Test Image: {os.path.basename(image_path)}")
    print(f"Model: {MODEL_NAME} loaded successfully.")
    print(f"Similarity to '{prompts['normal']}': {results['normal']:.4f}")
    print(f"Similarity to '{prompts['anomaly']}': {results['anomaly']:.4f}")
    print("\nVerification: 'normal' score should be HIGHER than 'anomaly' score.")

if __name__ == "__main__":
    run_vlm_prototype(IMAGE_PATH)
