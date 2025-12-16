"""
Prompt Engineering Module for CLIP Optimization
"""
import os
import json
import time
from typing import Dict

# ---------------------------------------------------------
# 1. SMART DICTIONARY (The "Brain" for your Demo)
# This maps keywords (and typos) to PERFECT CLIP PROMPTS.
# ---------------------------------------------------------
OFFLINE_KNOWLEDGE_BASE = {
    # SCENARIO A: FALLING / ACCIDENT
    "falling": "A photo of a person falling down, lying on the ground, or losing balance.",
    "fall": "A photo of a person falling down, lying on the ground, or losing balance.",
    "faling": "A photo of a person falling down, lying on the ground, or losing balance.", # Fixes 'faling'
    "lling": "A photo of a person falling down, lying on the ground, or losing balance.", # Fixes 'lling'
    "slip": "A photo of a person slipping and falling to the floor.",
    "slipping": "A photo of a person slipping and falling to the floor.",
    "crash": "A photo of a car crash or heavy vehicle collision.",
    "collision": "A photo of a vehicle collision or accident.",
    
    # SCENARIO B: NORMAL WALKING
    "walking": "A photo of a person walking normally in an upright posture.",
    "walk": "A photo of a person walking normally in an upright posture.",
    "wallking": "A photo of a person walking normally in an upright posture.", # Fixes 'wallking'
    "run": "A photo of a person running or jogging.",
    "running": "A photo of a person running or jogging.",
    "stand": "A photo of a person standing still.",
    "standing": "A photo of a person standing still.",
    "normal": "A photo of a person walking normally in an upright posture.",
    "perso": "A photo of a person walking normally in an upright posture.", # Fixes 'perso'
    "berson": "A photo of a person walking normally in an upright posture.", # Fixes 'berson'

    # SCENARIO C: EMPTY ROOM
    "empty": "A photo of an empty room with furniture but no people visible.",
    "emty": "A photo of an empty room with furniture but no people visible.",
    "nobody": "A photo of an empty room with furniture but no people visible."
}

def smart_offline_enhancer(text: str) -> str:
    """
    Fallback function that converts simple user inputs into 
    rich CLIP prompts using the knowledge base.
    Works offline - perfect for demos!
    """
    if not text or not text.strip():
        return "A photo showing a scene."
    
    text_lower = text.lower().strip()
    
    # Check for keywords in our dictionary (prioritize longer/more specific matches)
    # Sort by length (longest first) to match "falling" before "fall"
    sorted_keywords = sorted(OFFLINE_KNOWLEDGE_BASE.items(), key=lambda x: len(x[0]), reverse=True)
    
    for keyword, optimized_prompt in sorted_keywords:
        if keyword in text_lower:
            return optimized_prompt

    # If no keyword matches, fix common typos and create a basic prompt
    clean_text = text.replace("berson", "person").replace("perso", "person")
    clean_text = clean_text.replace("wallking", "walking").replace("walking", "walking")
    clean_text = clean_text.replace("lling", "falling").replace("faling", "falling")
    clean_text = clean_text.strip().rstrip(".")
    
    # If the text is very short or just a word, add context
    if len(clean_text.split()) <= 2:
        return f"A photo of {clean_text}, clearly visible in the frame."
    else:
        return f"A photo of {clean_text}."

# ---------------------------------------------------------
# 2. MAIN FUNCTION
# ---------------------------------------------------------
# Use available models (verified with this API key)
# gemini-2.5-flash is the latest and fastest available
MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-pro-latest",
    "gemini-1.5-flash-latest",
]

def get_optimized_prompts(raw_normal: str, raw_anomaly: str) -> Dict[str, str]:
    print("DEBUG: Enhancing prompts...") # Look for this in your terminal!
    
    # 1. Try to load the library
    try:
        import google.generativeai as genai
    except ImportError:
        print("DEBUG: Library missing, using offline mode.")
        return {
            "normal": smart_offline_enhancer(raw_normal),
            "anomaly": smart_offline_enhancer(raw_anomaly)
        }

    # 2. Check API Key - Try Streamlit secrets first, then environment variables
    api_key = None
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
            print(f"DEBUG: API key loaded from Streamlit secrets")
    except Exception as e:
        print(f"DEBUG: Failed to access Streamlit secrets: {e}")
    
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            print(f"DEBUG: API key loaded from environment variables")

    # If NO KEY, go straight to offline mode
    if not api_key:
        print("DEBUG: No API key found, using offline mode.")
        return {
            "normal": smart_offline_enhancer(raw_normal),
            "anomaly": smart_offline_enhancer(raw_anomaly)
        }

    try:
        import streamlit as st
        genai.configure(api_key=api_key)
        print(f"DEBUG: API key configured successfully")
    except Exception as e:
        print(f"DEBUG: Failed to configure API: {e}")
        return {
            "normal": smart_offline_enhancer(raw_normal),
            "anomaly": smart_offline_enhancer(raw_anomaly)
        }

    # 3. Try to call Google Generative AI
    full_prompt = f"""You are an expert at creating detailed, specific descriptions for image matching in computer vision.

Your task: Take user input and create rich, detailed CLIP prompts that describe scenarios clearly and specifically.

IMPORTANT - Use these examples as reference for the level of detail and specificity required:
- For falling/accident: "A photo of a person falling down, lying on the ground, or losing balance."
- For normal movement: "A photo of a person walking normally in an upright posture."
- For slipping: "A photo of a person slipping and falling to the floor."
- For running: "A photo of a person running or jogging."
- For crashes: "A photo of a car crash or heavy vehicle collision."
- For empty scenes: "A photo of an empty room with furniture but no people visible."

User inputs:
Normal scenario: '{raw_normal}'
Anomaly scenario: '{raw_anomaly}'

Create similarly detailed, specific prompts for these scenarios. Each prompt should:
1. Start with "A photo of" or similar
2. Include specific details about what's happening
3. Be 8-15 words long
4. Be suitable for image/video frame matching

Return ONLY valid JSON with 'normal' and 'anomaly' keys containing the detailed prompts (strings, not lists).
Example: {{"normal": "A photo of a person walking normally in an upright posture.", "anomaly": "A photo of a person falling down or lying on the ground."}}"""

    for model_name in MODEL_CANDIDATES:
        try:
            print(f"DEBUG: Attempting API call with model: {model_name}")
            model = genai.GenerativeModel(
                model_name,
                generation_config={
                    "temperature": 0.3,
                    "top_p": 0.95,
                    "response_mime_type": "application/json"
                }
            )
            response = model.generate_content(full_prompt)
            print(f"DEBUG: API response received")
            
            if response and response.text:
                result = json.loads(response.text)
                if "normal" in result and "anomaly" in result:
                    print(f"DEBUG: Successfully parsed API response")
                    # Handle both list and string responses
                    normal = result["normal"]
                    anomaly = result["anomaly"]
                    
                    # If responses are lists, take the first item
                    if isinstance(normal, list):
                        normal = normal[0] if normal else "A photo showing a normal scene."
                    if isinstance(anomaly, list):
                        anomaly = anomaly[0] if anomaly else "A photo showing an anomaly."
                    
                    return {"normal": normal, "anomaly": anomaly}
        except json.JSONDecodeError as e:
            print(f"DEBUG: JSON parsing error for {model_name}: {e}")
            continue
        except Exception as e:
            print(f"DEBUG: API call failed for {model_name}: {str(e)[:100]}")
            try:
                import streamlit as st
                st.warning(f"⚠️ Model {model_name} failed: {e}")
            except: pass
            continue

    # 4. FINAL FALLBACK (The Safety Net)
    print("DEBUG: API failed, using offline prompt enhancer")
    return {
        "normal": smart_offline_enhancer(raw_normal),
        "anomaly": smart_offline_enhancer(raw_anomaly)
    }