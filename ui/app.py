
import os
import sys
import tempfile
import time
from collections import deque

import altair as alt
import cv2
import pandas as pd
import streamlit as st
from PIL import Image
import torch

# Add parent directory to path to import ml_core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- AUDIO GUARD UTILS ---
AUDIO_KEYWORDS = [
    "audio", "sound", "noise", "loud", "quiet", "silence", "silent",
    "shout", "shouting", "scream", "screaming", "music", "mic",
    "microphone", "speaking", "speech", "voice", "voices",
]

def prompt_mentions_audio(prompt: str) -> bool:
    if not prompt: return False
    return any(keyword in prompt.lower() for keyword in AUDIO_KEYWORDS)

def guard_audio_prompts(prompt_normal: str, prompt_anomaly: str) -> bool:
    flagged = []
    if prompt_mentions_audio(prompt_normal): flagged.append("Normal Prompt")
    if prompt_mentions_audio(prompt_anomaly): flagged.append("Anomaly Prompt")
    
    if flagged:
        st.error(f"🔇 Audio cues not supported (found in {', '.join(flagged)}). Please describe visual events only.")
        return True
    return False

# --- CALLBACK FOR ENHANCE BUTTON ---
def handle_enhance_click():
    """
    This function runs BEFORE the app reloads.
    It updates the text boxes safely.
    """
    current_normal = st.session_state.input_normal_prompt
    current_anomaly = st.session_state.input_anomaly_prompt
    
    if not current_normal.strip() or not current_anomaly.strip():
        st.session_state["enhance_error"] = "Please provide both prompts before enhancing."
        return

    try:
        from ml_core.prompt_engineering import get_optimized_prompts
        
        # Show status while enhancing
        with st.spinner("🔄 Enhancing prompts..."):
            enhanced = get_optimized_prompts(current_normal, current_anomaly)
        
        st.session_state.input_normal_prompt = enhanced["normal"]
        st.session_state.input_anomaly_prompt = enhanced["anomaly"]
        st.session_state["show_enhance_success"] = True
        st.session_state["enhance_error"] = None
        
    except Exception as e:
        st.session_state["enhance_error"] = f"Enhancement failed: {str(e)}"
        st.session_state["show_enhance_success"] = False

# --- CORE FUNCTIONS ---

def get_analyze_video_function():
    from ml_core.video_analyzer import analyze_video
    return analyze_video

# --- LIVE STREAM LOGIC (OPTIMIZED) ---

def init_stream_state():
    """Initializes session state variables for the live stream."""
    if "stream_active" not in st.session_state:
        st.session_state.stream_active = False
    if "stream_cap" not in st.session_state:
        st.session_state.stream_cap = None
    if "data_buffers" not in st.session_state:
        # Use deques for rolling window data
        rolling_window = 60
        st.session_state.data_buffers = {
            "normal": deque(maxlen=rolling_window),
            "anomaly": deque(maxlen=rolling_window),
            "timestamps": deque(maxlen=rolling_window),
            "start_time": 0.0,
            "last_process_time": 0.0
        }
    # Cache for Model & Embeddings to prevent re-loading/re-computing every frame
    if "live_cache" not in st.session_state:
        st.session_state.live_cache = {
            "model": None,
            "preprocess": None,
            "device": None,
            "tokenizer": None,
            "text_features": None,
            "last_prompts": (None, None)
        }

def release_camera_safely():
    """Failsafe camera release."""
    if "stream_cap" in st.session_state and st.session_state.stream_cap is not None:
        st.session_state.stream_cap.release()
        st.session_state.stream_cap = None
    st.session_state.stream_active = False

def start_camera(camera_index):
    """Safely opens the camera and stores it in session state."""
    # Ensure any previous camera is closed
    release_camera_safely()

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        st.error(f"❌ Unable to access camera at index {camera_index}.")
        return False
        
    st.session_state.stream_cap = cap
    st.session_state.stream_active = True
    
    # Reset buffers
    rolling_window = 60
    st.session_state.data_buffers = {
        "normal": deque(maxlen=rolling_window),
        "anomaly": deque(maxlen=rolling_window),
        "timestamps": deque(maxlen=rolling_window),
        "start_time": time.time(),
        "last_process_time": 0.0
    }
    return True

def stop_camera():
    """Releases the camera and resets state."""
    release_camera_safely()

# FIX 3: Relax run_every to 0.04s (~25 FPS) to prevent UI thread overload
@st.fragment(run_every=0.04) 
def stream_widget(prompt_normal, prompt_anomaly, sampling_rate_fps):
    """
    Isolated UI fragment that handles the video feed and chart updates.
    Now includes caching for model and text embeddings.
    """
    if not st.session_state.stream_active or st.session_state.stream_cap is None:
        st.info("⏹️ Live monitoring stopped.")
        return

    # 1. Capture Frame
    cap = st.session_state.stream_cap
    ret, frame = cap.read()
    
    if not ret:
        st.error("⚠️ Lost connection to camera.")
        stop_camera()
        st.rerun()
        return

    # 2. Display Video (RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Layout within the fragment
    col_camera, col_results = st.columns([1, 2])
    
    with col_camera:
        st.subheader("📹 Live Feed")
        st.image(rgb_frame, channels="RGB", use_container_width=True)

    # 3. Process Logic (Throttled by sampling_rate)
    now = time.time()
    min_interval = 1.0 / max(0.1, sampling_rate_fps)
    last_time = st.session_state.data_buffers["last_process_time"]

    if now - last_time >= min_interval:
        st.session_state.data_buffers["last_process_time"] = now
        
        # --- FIX 1 & 2: SMART CACHING ---
        cache = st.session_state.live_cache
        current_prompts = (prompt_normal, prompt_anomaly)
        
        # Check if we need to load/reload resources
        if cache["model"] is None or cache["last_prompts"] != current_prompts:
            
            # Load Model (Only if not already loaded in session)
            if cache["model"] is None:
                from ml_core.vlm_test import get_model
                model, tokenizer, preprocess, device = get_model()
                # Explicitly set eval mode
                model.eval()
                
                cache["model"] = model
                cache["tokenizer"] = tokenizer
                cache["preprocess"] = preprocess
                cache["device"] = device
            
            # Compute Text Embeddings (Only if prompts changed)
            if cache["last_prompts"] != current_prompts:
                tokenizer = cache["tokenizer"]
                model = cache["model"]
                device = cache["device"]
                
                text_input = tokenizer([prompt_normal, prompt_anomaly]).to(device)
                with torch.no_grad():
                    text_features = model.encode_text(text_input)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                
                cache["text_features"] = text_features
                cache["last_prompts"] = current_prompts
        
        # Retrieve Resources from Cache
        model = cache["model"]
        preprocess = cache["preprocess"]
        device = cache["device"]
        text_features = cache["text_features"]

        from ml_core.video_processor import is_frame_valid

        # Check Validity
        if not is_frame_valid(rgb_frame):
            scores = [1.0, 0.0] # Default to Normal
        else:
            # Inference
            pil_image = Image.fromarray(rgb_frame)
            image_input = preprocess(pil_image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Use Cached Text Embeddings
                similarity = (image_features @ text_features.T).squeeze(0)
                scores = similarity.detach().cpu().numpy()

        # Update Buffers
        buffers = st.session_state.data_buffers
        buffers["normal"].append(float(scores[0]))
        buffers["anomaly"].append(float(scores[1]))
        buffers["timestamps"].append(now - buffers["start_time"])

    # 4. Render Charts (Always run to keep UI smooth)
    buffers = st.session_state.data_buffers
    if len(buffers["normal"]) > 1:
        # Compute scores on the fly
        from ml_core.anomaly_scorer import compute_anomaly_scores
        rolling_scores = compute_anomaly_scores(list(buffers["normal"]), list(buffers["anomaly"]))
        
        df = pd.DataFrame({
            "time_s": list(buffers["timestamps"]),
            "score": rolling_scores
        })
        
        latest_score = rolling_scores[-1]
        
        with col_results:
            st.subheader("📊 Live Analysis")
            
            # Try to get threshold from config, else default
            try:
                from ml_core.config import ANOMALY_THRESHOLD
            except ImportError:
                ANOMALY_THRESHOLD = 0.7

            # Metric
            delta = latest_score - 0.5
            st.metric(
                "Anomaly Score", 
                f"{latest_score:.3f}", 
                f"{delta:+.3f}",
                delta_color="inverse"
            )
            
            # Status
            if latest_score >= ANOMALY_THRESHOLD:
                st.error("🚨 **CRITICAL ANOMALY DETECTED**")
            elif latest_score > 0.5:
                st.warning("⚠️ Potential Anomaly")
            else:
                st.success("✅ Normal Activity")

            # Chart
            chart = alt.Chart(df).mark_line(color="#FF4B4B").encode(
                x=alt.X("time_s", axis=alt.Axis(title="Time (s)")),
                y=alt.Y("score", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(title="Anomaly Score"))
            ).properties(height=250)
            
            st.altair_chart(chart, use_container_width=True)


def render_video_upload_mode(uploaded_file, prompt_normal, prompt_anomaly, sampling_rate, start_requested):
    if uploaded_file:
        st.video(uploaded_file)
    
    if not start_requested: return
    if not uploaded_file: st.error("⚠️ Upload a video first."); return
    if guard_audio_prompts(prompt_normal, prompt_anomaly): return

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            uploaded_file.seek(0)
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        analyze_video = get_analyze_video_function()
        status_text.text("🔍 Analyzing...")
        progress_bar.progress(30)
        
        results = analyze_video(tmp_path, prompt_normal, prompt_anomaly, sampling_rate)
        os.unlink(tmp_path)
        progress_bar.progress(100)
        
        if results["status"] != "Success":
            st.error(f"❌ Error: {results.get('error_message')}")
            return

        meta = results["metadata"]
        st.success(f"✅ Analysis Complete. Max Score: {meta['max_anomaly_score']:.3f}")
        
        df = pd.DataFrame(results["data"])
        st.line_chart(df.set_index("time_s")["score"])

    except Exception as e:
        st.error(f"❌ Error: {e}")

# --- MAIN APP ---

def main():
    st.set_page_config(page_title="VLM Anomaly Detection", layout="wide")
    st.title("🎥 VLM-Powered Zero-Shot Video Anomaly Detection")

    # FIX 4: Failsafe Camera Release
    # If the user switches modes or reloads, this ensures the camera is freed
    # We check if the mode is changing or if the script is re-running from scratch
    if "stream_cap" in st.session_state and st.session_state.stream_cap is not None:
        # If we are NOT in live mode (checked below), we force release.
        pass 

    # Initialize State
    init_stream_state()
    if "input_normal_prompt" not in st.session_state: st.session_state["input_normal_prompt"] = "A person is walking normally."
    if "input_anomaly_prompt" not in st.session_state: st.session_state["input_anomaly_prompt"] = "A person is falling down."

    with st.sidebar:
        st.header("Mode Selection")
        mode = st.radio("Input Mode", ["Video File Upload", "Live Camera Feed"])
        
        # Enforce Camera Release on Mode Switch
        if mode == "Video File Upload" and st.session_state.stream_active:
            release_camera_safely()

        st.header("Prompts")
        prompt_normal = st.text_input("Normal Activity", key="input_normal_prompt")
        prompt_anomaly = st.text_input("Anomalous Event", key="input_anomaly_prompt")
        
        st.button("✨ Enhance Prompts", on_click=handle_enhance_click, help="Optimize prompts with AI")
        
        if st.session_state.get("show_enhance_success"):
            st.success("✅ Prompts enhanced!")
            st.session_state["show_enhance_success"] = False
            
        if st.session_state.get("enhance_error"):
            st.error(f"❌ {st.session_state['enhance_error']}")
            st.session_state["enhance_error"] = None

        if mode == "Video File Upload":
            st.header("Upload")
            uploaded_file = st.file_uploader("Choose Video", type=["mp4", "mov", "avi"])
            rate = st.slider("Sampling Rate", 0.5, 5.0, 1.0)
            start_btn = st.button("🚨 Start Analysis", type="primary")
        else:
            uploaded_file = None
            st.header("Live Stream")
            camera_idx = st.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1)
            rate = st.slider("Sampling Rate", 0.5, 5.0, 1.0, key="live_rate")
            
            # Start/Stop Buttons
            col1, col2 = st.columns(2)
            if col1.button("▶️ Start"):
                start_camera(camera_idx)
            if col2.button("⏹️ Stop"):
                stop_camera()

    # Logic Dispatch
    if mode == "Video File Upload":
        render_video_upload_mode(uploaded_file, prompt_normal, prompt_anomaly, rate, start_btn)
    else:
        # Calls the NON-BLOCKING fragment
        stream_widget(prompt_normal, prompt_anomaly, rate)

if __name__ == "__main__":
    main()
