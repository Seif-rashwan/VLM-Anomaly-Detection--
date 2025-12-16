import os
import sys
import tempfile
import time
import threading
import queue
from collections import deque

import altair as alt
import cv2
import pandas as pd
import streamlit as st
from PIL import Image
import torch

# Add parent directory to path to import ml_core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Configuration
from ml_core.config import (
    ANOMALY_THRESHOLD,
    ROLLING_WINDOW_SIZE,
    DEFAULT_SAMPLING_RATE
)

# Import Video Processing Utilities
from ml_core.video_processor import (
    is_frame_valid, 
    detect_motion,
    create_patches
)

# --- CONFIGURATION ---
FORCED_INFERENCE_INTERVAL = 3.0  # Seconds: Heartbeat for static anomalies
LOCALIZATION_COOLDOWN = 1.0      # Seconds: Throttle patch search to save FPS

# --- AUDIO GUARD UTILS ---
AUDIO_KEYWORDS = [
    "audio", "sound", "noise", "decibel", "volume", 
    "quiet", "silence", "silent", "music", 
    "mic", "microphone", "soundtrack", "mp3"
]

def prompt_mentions_audio(prompt: str) -> bool:
    if not prompt: return False
    return any(keyword in prompt.lower() for keyword in AUDIO_KEYWORDS)

def guard_audio_prompts(prompt_normal: str, prompt_anomaly: str) -> bool:
    flagged = []
    if prompt_mentions_audio(prompt_normal): flagged.append("Normal Prompt")
    if prompt_mentions_audio(prompt_anomaly): flagged.append("Anomaly Prompt")
    
    if flagged:
        st.error(
            f"⚠️ Visual Warning: The {', '.join(flagged)} contains audio-specific terms. "
            "Since this model only 'sees' video frames, try describing the *visual* action instead "
            "(e.g., use 'person yelling' instead of 'loud noise')."
        )
        return True
    return False

# --- CALLBACK FOR ENHANCE BUTTON ---
def handle_enhance_click():
    current_normal = st.session_state.input_normal_prompt
    current_anomaly = st.session_state.input_anomaly_prompt
    
    if not current_normal.strip() or not current_anomaly.strip():
        st.session_state["enhance_error"] = "Please provide both prompts before enhancing."
        return

    try:
        from ml_core.prompt_engineering import get_optimized_prompts
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

# --- THREADED CAMERA CLASS (Async Producer) ---
class ThreadedCamera:
    def __init__(self, src=0):
        if os.name == 'nt':
            self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.q = queue.Queue(maxsize=1)
        self.is_running = False
        self.thread = None
        self.last_frame_time = 0.0  # FIX 5: Track frame age
        
        if self.capture.isOpened():
            self.is_running = True
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()

    def _update(self):
        while self.is_running:
            ret, frame = self.capture.read()
            if not ret:
                self.is_running = False
                break
            # FIX 5: Timestamp the frame at capture time
            self.last_frame_time = time.time()
            
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            try:
                self.q.put(frame, timeout=0.01)
            except queue.Full:
                pass

    def read(self):
        try:
            return True, self.q.get(timeout=0.1) 
        except queue.Empty:
            return False, None

    def isOpened(self):
        return self.capture.isOpened()

    def release(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.capture.release()

# --- LIVE STREAM LOGIC ---

def init_stream_state():
    """Initializes session state variables."""
    if "stream_active" not in st.session_state:
        st.session_state.stream_active = False
    if "stream_cap" not in st.session_state:
        st.session_state.stream_cap = None
        
    if "data_buffers" not in st.session_state:
        st.session_state.data_buffers = {
            "normal": deque(maxlen=ROLLING_WINDOW_SIZE),
            "anomaly": deque(maxlen=ROLLING_WINDOW_SIZE),
            "timestamps": deque(maxlen=ROLLING_WINDOW_SIZE),
            "start_time": 0.0,
            "last_process_time": 0.0,
            "last_inference_time": 0.0,
            "last_localization_time": 0.0  # FIX 4: Throttle localization
        }

    if "prev_frame" not in st.session_state:
        st.session_state.prev_frame = None

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
    if "stream_cap" in st.session_state and st.session_state.stream_cap is not None:
        st.session_state.stream_cap.release()
        st.session_state.stream_cap = None
    st.session_state.stream_active = False

def start_camera(camera_index):
    release_camera_safely()
    cap = ThreadedCamera(camera_index)
    
    if not cap.isOpened():
        st.error(f"❌ Unable to access camera at index {camera_index}.")
        return False
        
    st.session_state.stream_cap = cap
    st.session_state.stream_active = True
    
    st.session_state.data_buffers = {
        "normal": deque(maxlen=ROLLING_WINDOW_SIZE),
        "anomaly": deque(maxlen=ROLLING_WINDOW_SIZE),
        "timestamps": deque(maxlen=ROLLING_WINDOW_SIZE),
        "start_time": time.time(),
        "last_process_time": 0.0,
        "last_inference_time": 0.0,
        "last_localization_time": 0.0
    }
    st.session_state.prev_frame = None
    return True

def stop_camera():
    release_camera_safely()

@st.fragment(run_every=0.04) 
def stream_widget(prompt_normal, prompt_anomaly, sampling_rate_fps):
    if not st.session_state.stream_active or st.session_state.stream_cap is None:
        st.info("⏹️ Live monitoring stopped.")
        return

    cap = st.session_state.stream_cap
    ret, frame = cap.read()
    
    if not ret:
        st.error("⚠️ Lost connection to camera.")
        stop_camera()
        st.rerun()
        return

    # FIX 5: Freshness Check
    # If the frame is stale (>0.5s old), drop it to prevent ghosting
    frame_age = time.time() - cap.last_frame_time
    if frame_age > 0.5:
        return

    # --- 1. MOTION DETECTION (CV2) ---
    prev_frame = st.session_state.get("prev_frame", None)
    motion_found, contours, current_processed_frame = detect_motion(frame, prev_frame)
    st.session_state.prev_frame = current_processed_frame

    # Create Clean Frame for CLIP & Display Frame for UI
    clean_rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    display_frame = clean_rgb_frame.copy() 

    # Draw Motion Highlights (Red Boxes)
    if motion_found:
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    col_camera, col_results = st.columns([1, 2])
    with col_camera:
        st.subheader("📹 Live Feed") 

    # --- 2. LOGIC GATE & INFERENCE ---
    now = time.time()
    min_interval = 1.0 / max(0.1, sampling_rate_fps)
    last_process_time = st.session_state.data_buffers["last_process_time"]
    last_inference_time = st.session_state.data_buffers.get("last_inference_time", 0.0)

    if now - last_process_time >= min_interval:
        st.session_state.data_buffers["last_process_time"] = now
        
        # FIX 2: Priority Logic
        # Run if Motion is detected OR (No Motion AND Heartbeat interval exceeded)
        should_run_clip = motion_found or (
            not motion_found and (now - last_inference_time >= FORCED_INFERENCE_INTERVAL)
        )
        
        if should_run_clip:
            st.session_state.data_buffers["last_inference_time"] = now
            
            cache = st.session_state.live_cache
            current_prompts = (prompt_normal, prompt_anomaly)
            
            # Load resources if needed
            if cache["model"] is None or cache["last_prompts"] != current_prompts:
                if cache["model"] is None:
                    from ml_core.vlm_test import get_model
                    model, tokenizer, preprocess, device = get_model()
                    model.eval()
                    cache["model"] = model
                    cache["tokenizer"] = tokenizer
                    cache["preprocess"] = preprocess
                    cache["device"] = device
                
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
            
            model = cache["model"]
            preprocess = cache["preprocess"]
            device = cache["device"]
            text_features = cache["text_features"]

            # Run Inference on CLEAN FRAME
            if not is_frame_valid(clean_rgb_frame):
                scores = [0.9, 0.1] # FIX 3: Soft Normal for invalid frames
            else:
                pil_image = Image.fromarray(clean_rgb_frame)
                image_input = preprocess(pil_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = model.encode_image(image_input)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    
                    logits = (image_features @ text_features.T)
                    
                    # FIX 1: Use Probabilities, not Logits
                    probs = logits.softmax(dim=-1)[0]
                    scores = probs.detach().cpu().numpy() # [Normal_Prob, Anomaly_Prob]
                    global_prob = float(scores[1])

                    # FIX 4: Throttled Anomaly Localization
                    last_loc_time = st.session_state.data_buffers.get("last_localization_time", 0.0)
                    
                    if global_prob > ANOMALY_THRESHOLD and (now - last_loc_time > LOCALIZATION_COOLDOWN):
                        # Update timestamp
                        st.session_state.data_buffers["last_localization_time"] = now
                        
                        patch_batch, coords = create_patches(torch.from_numpy(clean_rgb_frame).permute(2, 0, 1), grid_size=4)
                        patch_batch = patch_batch.to(device)
                        
                        p_features = model.encode_image(patch_batch)
                        p_features /= p_features.norm(dim=-1, keepdim=True)
                        
                        p_probs = (p_features @ text_features.T).softmax(dim=-1)[:, 1]
                        best_idx = torch.argmax(p_probs).item()
                        
                        if p_probs[best_idx] > ANOMALY_THRESHOLD:
                            st.session_state.current_bbox = coords[best_idx]
                        else:
                            st.session_state.current_bbox = None
                    
                    # If global probability drops, clear the box immediately
                    elif global_prob <= ANOMALY_THRESHOLD:
                         st.session_state.current_bbox = None

        else:
            # FIX 3: Soft Normal Assumption
            # If we skip CLIP, we assume the scene is mostly normal, but allow small uncertainty
            scores = [0.9, 0.1] 
            st.session_state.current_bbox = None

        buffers = st.session_state.data_buffers
        buffers["normal"].append(float(scores[0]))
        buffers["anomaly"].append(float(scores[1]))
        buffers["timestamps"].append(now - buffers["start_time"])

    # Draw Anomaly Box (Blue)
    if st.session_state.get("current_bbox"):
        x, y, w, h = st.session_state.current_bbox
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 3) 
        cv2.putText(display_frame, "ANOMALY", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Render Display
    with col_camera:
        st.image(display_frame, channels="RGB", width="stretch")

    buffers = st.session_state.data_buffers
    if len(buffers["normal"]) > 1:
        from ml_core.anomaly_scorer import compute_anomaly_scores
        rolling_scores = compute_anomaly_scores(list(buffers["normal"]), list(buffers["anomaly"]))
        
        df = pd.DataFrame({
            "time_s": list(buffers["timestamps"]),
            "score": rolling_scores
        })
        
        latest_score = rolling_scores[-1]
        
        with col_results:
            st.subheader("📊 Live Analysis")
            delta = latest_score - 0.5
            st.metric("Anomaly Score", f"{latest_score:.3f}", f"{delta:+.3f}", delta_color="inverse")
            
            if latest_score >= ANOMALY_THRESHOLD:
                st.error("🚨 **CRITICAL ANOMALY DETECTED**")
            elif latest_score > 0.5:
                st.warning("⚠️ Potential Anomaly")
            else:
                st.success("✅ Normal Activity")

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

def main():
    st.set_page_config(page_title="VLM Anomaly Detection", layout="wide")
    st.title("🎥 VLM-Powered Zero-Shot Video Anomaly Detection")

    if "stream_cap" in st.session_state and st.session_state.stream_cap is not None:
        pass 

    init_stream_state()
    if "input_normal_prompt" not in st.session_state: st.session_state["input_normal_prompt"] = "A person is walking normally."
    if "input_anomaly_prompt" not in st.session_state: st.session_state["input_anomaly_prompt"] = "A person is falling down."

    with st.sidebar:
        st.header("Mode Selection")
        mode = st.radio("Input Mode", ["Video File Upload", "Live Camera Feed"])
        
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
            # REFACTORED: Use constant from config
            rate = st.slider("Sampling Rate", 0.5, 5.0, DEFAULT_SAMPLING_RATE)
            start_btn = st.button("🚨 Start Analysis", type="primary")
        else:
            uploaded_file = None
            st.header("Live Stream")
            camera_idx = st.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1)
            # REFACTORED: Use constant from config
            rate = st.slider("Sampling Rate", 0.5, 5.0, DEFAULT_SAMPLING_RATE, key="live_rate")
            
            col1, col2 = st.columns(2)
            if col1.button("▶️ Start"):
                start_camera(camera_idx)
            if col2.button("⏹️ Stop"):
                stop_camera()

    if mode == "Video File Upload":
        render_video_upload_mode(uploaded_file, prompt_normal, prompt_anomaly, rate, start_btn)
    else:
        stream_widget(prompt_normal, prompt_anomaly, rate)

if __name__ == "__main__":
    main()