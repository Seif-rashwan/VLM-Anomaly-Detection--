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
MODEL_CANDIDATES = ["gemini-1.5-flash", "gemini-pro"]

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

    # 2. Check API Key
    api_key = None
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
    except:
        pass
    
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")

    # If NO KEY, go straight to offline mode
    if not api_key:
        print("DEBUG: No API key found, using offline mode.")
        return {
            "normal": smart_offline_enhancer(raw_normal),
            "anomaly": smart_offline_enhancer(raw_anomaly)
        }

    genai.configure(api_key=api_key)

    # 3. Try to call Google (likely to fail with 429)
    full_prompt = f"Fix typos and describe for CLIP:\nNormal: '{raw_normal}'\nAnomaly: '{raw_anomaly}'"

    for model_name in MODEL_CANDIDATES:
        try:
            model = genai.GenerativeModel(model_name, generation_config={"response_mime_type": "application/json"})
            response = model.generate_content(full_prompt)
            result = json.loads(response.text)
            if "normal" in result:
                return {"normal": result["normal"], "anomaly": result["anomaly"]}
        except Exception as e:
            # print(f"DEBUG: API failed ({e})") 
            continue

    # 4. FINAL FALLBACK (The Safety Net)
    print("⚠️ API Failed (Quota/Error). Using Smart Offline Enhancer.")
    return {
        "normal": smart_offline_enhancer(raw_normal),
        "anomaly": smart_offline_enhancer(raw_anomaly)
    }