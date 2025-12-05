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

# Add parent directory to path to import ml_core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AUDIO_KEYWORDS = [
    "audio",
    "sound",
    "noise",
    "loud",
    "quiet",
    "silence",
    "silent",
    "shout",
    "shouting",
    "scream",
    "screaming",
    "music",
    "mic",
    "microphone",
    "speaking",
    "speech",
    "voice",
    "voices",
]


def prompt_mentions_audio(prompt: str) -> bool:
    """Return True if the prompt references audio-based cues."""
    if not prompt:
        return False

    normalized = prompt.lower()
    return any(keyword in normalized for keyword in AUDIO_KEYWORDS)


def guard_audio_prompts(prompt_normal: str, prompt_anomaly: str) -> bool:
    """
    Warn the user if they describe purely audio cues (unsupported).
    Returns True when audio terms are detected so callers can abort processing.
    """
    flagged_prompts = []

    if prompt_mentions_audio(prompt_normal):
        flagged_prompts.append("Normal Activity Prompt")
    if prompt_mentions_audio(prompt_anomaly):
        flagged_prompts.append("Anomalous Event Prompt")

    if not flagged_prompts:
        return False

    st.error(
        "🔇 Audio-based cues are not supported. Please describe **visual** events only "
        f"(check: {', '.join(flagged_prompts)})."
    )
    st.caption(
        "Tip: Rephrase prompts to reference what the camera can see (e.g., gestures, actions, objects)."
    )
    return True

# Lazily import heavy modules/functions when needed
def get_analyze_video_function():
    """Lazy import of the offline video analysis function."""
    from ml_core.video_analyzer import analyze_video

    return analyze_video


def analyze_realtime_stream(
    camera_index: int,
    prompt_normal: str,
    prompt_anomaly: str,
    sampling_rate_fps: float = 1.0,
):
    """
    Stream video from a camera, score frames in real time, and update Streamlit UI.
    Keeps a rolling window of the latest 60 sampled scores/timestamps.
    """
    from ml_core.vlm_test import get_model
    from ml_core.anomaly_scorer import compute_anomaly_scores
    import torch  # Local import to avoid loading at module import time

    st.info("Initializing live camera stream... Please allow access if prompted.")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        st.error("❌ Unable to access the selected camera.")
        return
    
    # Optimize camera settings for lower latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize delay
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set reasonable FPS

    try:
        model, tokenizer, preprocess, device = get_model()
        prompts = [prompt_normal, prompt_anomaly]
        text_input = tokenizer(prompts).to(device)

        with torch.no_grad():
            text_features = model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        rolling_window = 60
        normal_scores = deque(maxlen=rolling_window)
        anomaly_scores = deque(maxlen=rolling_window)
        timestamps = deque(maxlen=rolling_window)

        # Create a two-column layout: small camera feed on left, results on right
        col_camera, col_results = st.columns([1, 2])

        with col_camera:
            st.subheader("📹 Live Camera Feed")
            frame_placeholder = st.empty()

        with col_results:
            chart_placeholder = st.empty()
            metrics_placeholder = st.empty()
            status_placeholder = st.empty()
            table_placeholder = st.empty()

        sampling_interval = 1.0 / max(0.1, sampling_rate_fps)
        last_sample_time = 0.0
        last_ui_update = 0.0
        ui_update_interval = 0.5  # Update UI every 0.5 seconds to reduce overhead
        stream_start = time.time()

        with col_results:
            status_placeholder.info("📡 Live monitoring started. Press the stop button to end the stream.")

        while True:
            # Check if stream should stop
            if not st.session_state.get("live_stream_requested", False):
                with col_results:
                    status_placeholder.info("⏹️ Live monitoring stopped.")
                break

            # Read frame and skip buffered frames to get the latest (reduces latency)
            ret, frame = cap.read()
            if not ret:
                with col_results:
                    status_placeholder.error("⚠️ Lost connection to the camera.")
                break
            
            # Skip a few buffered frames to get the most recent frame
            for _ in range(2):
                ret_new, frame_new = cap.read()
                if ret_new:
                    ret, frame = ret_new, frame_new
                else:
                    break

            # Always display the camera feed (even if not analyzing this frame)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with col_camera:
                frame_placeholder.image(rgb_frame, caption="Live Feed", use_container_width=True)

            now = time.time()
            
            # Only process frame if sampling interval has passed
            if now - last_sample_time < sampling_interval:
                time.sleep(0.01)  # Small sleep to prevent CPU spinning
                continue

            last_sample_time = now
            pil_image = Image.fromarray(rgb_frame)

            # Process frame with model (this is the expensive operation)
            image_input = preprocess(pil_image).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                similarity = (image_features @ text_features.T).squeeze(0)
                similarity_scores = similarity.detach().cpu().numpy()

            normal_scores.append(float(similarity_scores[0]))
            anomaly_scores.append(float(similarity_scores[1]))
            timestamps.append(now - stream_start)

            if len(normal_scores) == 0:
                continue

            # Compute scores
            rolling_anomaly_scores = compute_anomaly_scores(
                list(normal_scores), list(anomaly_scores)
            )
            latest_score = rolling_anomaly_scores[-1]

            # Only update heavy UI elements (chart/table) periodically to reduce lag
            if now - last_ui_update >= ui_update_interval or len(normal_scores) == 1:
                last_ui_update = now
                
                df = pd.DataFrame(
                    {
                        "time_s": list(timestamps),
                        "score": rolling_anomaly_scores,
                    }
                )

                # Update results in right column
                with col_results:
                    chart = (
                        alt.Chart(df)
                        .mark_line(color="#FF4B4B")
                        .encode(
                            x=alt.X("time_s", title="Time (s)"),
                            y=alt.Y("score", scale=alt.Scale(domain=[0, 1]), title="Anomaly Score"),
                        )
                    )
                    threshold_line = alt.Chart(pd.DataFrame({"score": [0.5]})).mark_rule(color="orange").encode(y="score")

                    chart_placeholder.altair_chart(chart + threshold_line, use_container_width=True)

                    table_placeholder.dataframe(df, use_container_width=True, height=250)

            # Always update metrics and status (lightweight)
            with col_results:
                delta = latest_score - 0.5
                metrics_placeholder.metric(
                    label="Latest Anomaly Score",
                    value=f"{latest_score:.3f}",
                    delta=f"{delta:+.3f} vs threshold",
                )

                state_message = (
                    "🚨 Anomalous pattern detected." if latest_score > 0.5 else "✅ Scene is within normal range."
                )
                status_placeholder.markdown(state_message)

    finally:
        cap.release()


def render_video_upload_mode(
    uploaded_file,
    prompt_normal: str,
    prompt_anomaly: str,
    sampling_rate: float,
    start_requested: bool,
):
    """Handle the offline video upload workflow."""
    if uploaded_file is not None:
        st.subheader("📹 Video Preview")
        uploaded_file.seek(0)
        st.video(uploaded_file, format="video/mp4")
        st.caption(f"Video: {uploaded_file.name}")

    if not start_requested:
        return

    if uploaded_file is None:
        st.error("⚠️ Please upload a video file to start analysis.")
        return

    if not prompt_normal.strip() or not prompt_anomaly.strip():
        st.error("⚠️ Please provide both normal and anomaly prompts.")
        return

    if guard_audio_prompts(prompt_normal, prompt_anomaly):
        return

    progress_bar = st.progress(0)
    status_text = st.empty()

    with st.spinner("🔄 Processing video with VLM model..."):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                uploaded_file.seek(0)
                tmp.write(uploaded_file.read())
                tmp_video_path = tmp.name

            status_text.text("📥 Video loaded. Processing frames...")
            progress_bar.progress(15)

            analyze_video = get_analyze_video_function()

            status_text.text("🔍 Analyzing frames with VLM...")
            progress_bar.progress(35)

            results = analyze_video(
                video_path=tmp_video_path,
                prompt_normal=prompt_normal,
                prompt_anomaly=prompt_anomaly,
                sampling_rate_fps=sampling_rate,
            )

            progress_bar.progress(90)
            status_text.text("📊 Computing anomaly scores...")

            os.unlink(tmp_video_path)

            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()

            if results["status"] != "Success":
                st.error(f"❌ Analysis failed: {results.get('error_message', 'Unknown error')}")
                return

            st.success("✅ Analysis complete!")
            metadata = results["metadata"]
            max_anomaly_time = metadata.get("max_anomaly_time", 0.0)
            max_anomaly_score = metadata.get("max_anomaly_score", 0.0)
            minutes = int(max_anomaly_time // 60)
            seconds = int(max_anomaly_time % 60)
            timestamp_str = f"{minutes}:{seconds:02d}"

            if max_anomaly_score > 0.5:
                st.warning(f"🚨 Anomaly detected at {timestamp_str} (Score: {max_anomaly_score:.3f})")
            else:
                st.info(f"ℹ️ Highest score {max_anomaly_score:.3f} at {timestamp_str}. No strong anomaly detected.")

            st.subheader("📊 Analysis Summary")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Frames", metadata.get("total_frames", 0))
            col2.metric("Duration", f"{metadata.get('total_seconds', 0):.1f}s")
            col3.metric("Max Score", f"{metadata.get('max_anomaly_score', 0):.3f}")
            col4.metric("Peak Time", timestamp_str)

            st.subheader("📈 Anomaly Score Over Time")
            if len(results["data"]) > 0:
                df = pd.DataFrame(results["data"])
                st.line_chart(df.set_index("time_s")["score"], use_container_width=True, height=400)
                st.caption("Score range: 0 (normal) to 1 (anomalous). Threshold: 0.5")
                with st.expander("📋 View Raw Data"):
                    st.dataframe(df, use_container_width=True)
            else:
                st.warning("No data points to display.")

            with st.expander("📋 Detailed Results"):
                st.write(f"**Normal Prompt:** {prompt_normal}")
                st.write(f"**Anomaly Prompt:** {prompt_anomaly}")
                st.write(f"**Sampling Rate:** {sampling_rate} fps")
                st.write(f"**Average Score:** {metadata.get('average_score', 0):.4f}")
                st.write(f"**Score Range:** {metadata.get('min_score', 0):.4f} - {metadata.get('max_score', 0):.4f}")
                st.write(f"**Std Dev:** {metadata.get('std_score', 0):.4f}")

        except Exception as exc:
            st.error(f"❌ Error during analysis: {exc}")
            st.exception(exc)
        finally:
            if "tmp_video_path" in locals() and os.path.exists(tmp_video_path):
                os.unlink(tmp_video_path)


def main():
    st.set_page_config(page_title="VLM Video Anomaly Detection", layout="wide")

    st.title("🎥 VLM-Powered Zero-Shot Video Anomaly Detection")
    st.markdown(
        "Monitor anomalies via uploaded footage or a live camera stream using CLIP (ViT-B-16) scoring."
    )

    if "live_stream_requested" not in st.session_state:
        st.session_state["live_stream_requested"] = False

    with st.sidebar:
        st.header("Mode Selection")
        mode = st.radio(
            "Choose Input Mode",
            options=["Video File Upload", "Live Camera Feed"],
            index=0,
        )

        st.header("Prompts")
        prompt_normal = st.text_input(
            "Normal Activity Prompt",
            value="A person is walking normally.",
            placeholder="Describe the expected, normal behavior.",
        )
        prompt_anomaly = st.text_input(
            "Anomalous Event Prompt",
            value="A person is falling down.",
            placeholder="Describe the event you consider anomalous.",
        )

        if mode == "Video File Upload":
            st.header("Video Upload")
            uploaded_file = st.file_uploader(
                "Choose a video file (MP4, MOV, AVI, etc.)",
                type=["mp4", "mov", "avi", "mkv", "webm"],
            )

            st.header("Sampling")
            sampling_rate = st.slider(
                "Sampling Rate (fps)",
                min_value=0.5,
                max_value=5.0,
                value=1.0,
                step=0.5,
                help="Higher values = more frames analyzed per second.",
            )

            start_video_analysis = st.button("🚨 Start Anomaly Analysis", type="primary")
        else:
            uploaded_file = None
            st.header("Live Stream Settings")
            sampling_rate = st.slider(
                "Sampling Rate (fps)",
                min_value=0.5,
                max_value=5.0,
                value=1.0,
                step=0.5,
                help="Controls how often frames from the camera are analyzed.",
                key="live_sampling_slider",
            )

            start_video_analysis = False
            start_live = st.button("▶️ Start Live Monitoring", type="primary")
            stop_live = st.button("⏹️ Stop Live Monitoring")

            if start_live:
                if not guard_audio_prompts(prompt_normal, prompt_anomaly):
                    st.session_state["live_stream_requested"] = True
            if stop_live:
                st.session_state["live_stream_requested"] = False

    if mode == "Video File Upload":
        render_video_upload_mode(
            uploaded_file=uploaded_file,
            prompt_normal=prompt_normal,
            prompt_anomaly=prompt_anomaly,
            sampling_rate=sampling_rate,
            start_requested=start_video_analysis,
        )
    else:
        if st.session_state.get("live_stream_requested", False):
            analyze_realtime_stream(
                camera_index=0,
                prompt_normal=prompt_normal,
                prompt_anomaly=prompt_anomaly,
                sampling_rate_fps=sampling_rate,
            )
        else:
            st.info("Press **▶️ Start Live Monitoring** in the sidebar to begin streaming.")


if __name__ == "__main__":
    main()
