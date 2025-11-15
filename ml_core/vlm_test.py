import torch
import open_clip
from PIL import Image
import os
import numpy as np

# --- Configuration ---
# Use a smaller, efficient model for testing (e.g., ViT-B/16)
MODEL_NAME = "ViT-B-16"
PRETRAINED_WEIGHTS = "laion2b_s34b_b88k"
IMAGE_PATH = os.path.join("..", "data", "test_image.jpg")

# --- Core Logic ---

def run_vlm_prototype(image_path: str):
    """
    Loads CLIP, processes an image and two prompts, and calculates similarity scores.
    """
    if not os.path.exists(image_path):
        print(f"Error: Test image not found at {image_path}. Please place a file there.")
        return

    # 1. Load the Model and Preprocessor
    # Check if GPU is available (Colab will use this)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading VLM on device: {device}")
    
    # Load the model, tokenizer, and image preprocessor
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, 
        pretrained=PRETRAINED_WEIGHTS, 
        device=device
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    # 2. Define Test Prompts
    # Assume the test image isa car
    prompts = {
        "normal": "car parked safely outdoors",
        "anomaly": "A heavily damaged car after a crash." 
    }

    # 3. Process Image and Text
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device) # Add batch dimension
    
    # Tokenize the text prompts
    text_input = tokenizer(list(prompts.values())).to(device)

    # 4. Calculate Embeddings (Features)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)

    # Normalize features for similarity calculation (crucial for CLIP)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # 5. Calculate Similarity Score (Dot Product / Cosine Similarity)
    # The dot product of normalized vectors gives the cosine similarity
    similarity = (image_features @ text_features.T).squeeze(0)
    
    # --- Output and Verification ---
    
    print("\n--- VLM Prototype Results ---")
    print(f"Test Image: {os.path.basename(image_path)}")
    print(f"Model: {MODEL_NAME} loaded successfully.")
    
    results = dict(zip(prompts.keys(), similarity.cpu().numpy()))

    print(f"Similarity to '{prompts['normal']}': {results['normal']:.4f}")
    print(f"Similarity to '{prompts['anomaly']}': {results['anomaly']:.4f}")
    print("\nVerification: 'normal' score should be HIGHER than 'anomaly' score.")

if __name__ == "__main__":
    run_vlm_prototype(IMAGE_PATH)
