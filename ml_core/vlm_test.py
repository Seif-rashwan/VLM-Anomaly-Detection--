import torch
import open_clip
from PIL import Image
import os
import numpy as np
from typing import Union, Dict
import warnings
import sys
from io import StringIO
from ml_core.config import MODEL_NAME, PRETRAINED_WEIGHTS

# Suppress PyTorch torch.classes warning (harmless but noisy)
# Do this at module level to catch all instances
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch.classes.*")

# Suppress at stderr level too
old_stderr = sys.stderr
sys.stderr = StringIO()

# Global model cache
_model_cache = None
_tokenizer_cache = None
_preprocess_cache = None
_device_cache = None

def get_model():
    """Load and cache the model to avoid reloading on each call."""
    global _model_cache, _tokenizer_cache, _preprocess_cache, _device_cache
    
    if _model_cache is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            # Suppress all warnings during model loading
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model, _, preprocess = open_clip.create_model_and_transforms(
                    MODEL_NAME, 
                    pretrained=PRETRAINED_WEIGHTS, 
                    device=device
                )
        finally:
            # Restore stderr
            sys.stderr = old_stderr
        model.eval() # Ensure eval mode is set immediately
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        
        _model_cache = model
        _tokenizer_cache = tokenizer
        _preprocess_cache = preprocess
        _device_cache = device
    
    return _model_cache, _tokenizer_cache, _preprocess_cache, _device_cache

def analyze_image_vlm(
    image: Union[str, Image.Image], 
    normal_prompt: str, 
    anomaly_prompt: str
) -> Dict[str, float]:
    """Analyzes an image using VLM (CLIP)."""
    model, tokenizer, preprocess, device = get_model()
    
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image not found at {image}")
        pil_image = Image.open(image).convert("RGB")
    else:
        pil_image = image.convert("RGB") if image.mode != "RGB" else image
    
    image_input = preprocess(pil_image).unsqueeze(0).to(device)
    prompts = [normal_prompt, anomaly_prompt]
    text_input = tokenizer(prompts).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
    
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    similarity = (image_features @ text_features.T).squeeze(0)
    similarity_scores = similarity.cpu().numpy()
    
    return {
        "normal": float(similarity_scores[0]),
        "anomaly": float(similarity_scores[1])
    }

if __name__ == "__main__":
    # Test block
    print("Model loader test...")
    get_model()
    print("Model loaded successfully.")