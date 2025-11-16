import streamlit as st
import os
import sys
import tempfile
import pandas as pd

# Add parent directory to path to import ml_core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Lazy import to avoid loading PyTorch at module level (prevents Streamlit file watcher warnings)
def get_analyze_video_function():
    """Lazy import of the VLM video analysis function."""
    from ml_core.video_analyzer import analyze_video
    return analyze_video

# Set page configuration for a professional look
st.set_page_config(
    page_title="VLM Video Anomaly Detection",
    layout="wide"
)

# --- Application Layout ---

def render_ui_shell():
    """Renders the Streamlit UI for VLM-based video anomaly detection."""
    
    st.title("📹 VLM-Powered Zero-Shot Video Anomaly Detection")
    st.markdown("Upload a video and define the normal and anomalous conditions using natural language prompts.")

    # --- Sidebar for Inputs ---
    with st.sidebar:
        st.header("1. Upload Video")
        # Widget for video file upload
        uploaded_file = st.file_uploader(
            "Choose a video file (MP4, MOV, AVI, etc.)", 
            type=['mp4', 'mov', 'avi', 'mkv', 'webm']
        )
        
        st.header("2. Define Conditions")
        # Two text input boxes for the user's natural language prompts
        normal_prompt = st.text_input(
            "Normal Activity Prompt (P_Normal)", 
            value="A person is walking normally.",
            placeholder="e.g., A person cycling on the road."
        )
        anomaly_prompt = st.text_input(
            "Anomalous Event Prompt (P_Anomaly)", 
            value="A person is falling down.",
            placeholder="e.g., A traffic violation or car crash."
        )
        
        st.header("3. Analysis Settings")
        sampling_rate = st.slider(
            "Sampling Rate (frames per second)",
            min_value=0.5,
            max_value=5.0,
            value=1.0,
            step=0.5,
            help="Lower values = faster processing, less detail. Higher values = slower processing, more detail."
        )

        # The core action button
        start_analysis_button = st.button("🚨 Start Anomaly Analysis", type="primary")

    # --- Main Content Area ---
    
    # Display uploaded video
    if uploaded_file is not None:
        st.subheader("📹 Video Preview")
        
        # Display video (reset file pointer first)
        uploaded_file.seek(0)
        st.video(uploaded_file, format="video/mp4")
        st.caption(f"Video: {uploaded_file.name}")
    
    # Analysis Results Section
    if start_analysis_button:
        if uploaded_file is None:
            st.error("⚠️ Please upload a video file to start analysis.")
        elif not normal_prompt.strip() or not anomaly_prompt.strip():
            st.error("⚠️ Please provide both normal and anomaly prompts.")
        else:
            # Show loading spinner
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("🔄 Processing video with VLM model... This may take several minutes depending on video length."):
                try:
                    # Save uploaded file to temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        uploaded_file.seek(0)
                        tmp_file.write(uploaded_file.read())
                        tmp_video_path = tmp_file.name
                    
                    status_text.text("📥 Video loaded. Processing frames...")
                    progress_bar.progress(10)
                    
                    # Lazy load the analysis function
                    analyze_video = get_analyze_video_function()
                    
                    # Run VLM analysis
                    status_text.text("🔍 Analyzing frames with VLM...")
                    progress_bar.progress(30)
                    
                    results = analyze_video(
                        video_path=tmp_video_path,
                        prompt_normal=normal_prompt,
                        prompt_anomaly=anomaly_prompt,
                        sampling_rate_fps=sampling_rate
                    )
                    
                    progress_bar.progress(90)
                    status_text.text("📊 Computing anomaly scores...")
                    
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_video_path)
                    except:
                        pass
                    
                    progress_bar.progress(100)
                    status_text.empty()
                    progress_bar.empty()
                    
                    # Display results
                    if results["status"] == "Success":
                        st.success("✅ Analysis complete!")
                        
                        # Display metadata
                        metadata = results["metadata"]
                        
                        # Phase 3 Requirement: Prominently highlight anomaly timestamp
                        max_anomaly_time = metadata.get('max_anomaly_time', 0)
                        max_anomaly_score = metadata.get('max_anomaly_score', 0)
                        
                        # Format time as MM:SS
                        minutes = int(max_anomaly_time // 60)
                        seconds = int(max_anomaly_time % 60)
                        time_str = f"{minutes}:{seconds:02d}"
                        
                        # Display prominent anomaly detection message
                        if max_anomaly_score > 0.5:  # Threshold for anomaly
                            st.warning(f"🚨 **Anomaly detected at {time_str}** (Score: {max_anomaly_score:.3f})")
                        else:
                            st.info(f"ℹ️ **No significant anomaly detected.** Highest score: {max_anomaly_score:.3f} at {time_str}")
                        
                        st.subheader("📊 Analysis Summary")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Frames", metadata.get("total_frames", 0))
                        with col2:
                            st.metric("Duration", f"{metadata.get('total_seconds', 0):.1f}s")
                        with col3:
                            st.metric("Max Anomaly Score", f"{max_anomaly_score:.3f}")
                        with col4:
                            st.metric("Peak Anomaly Time", f"{time_str}")
                        
                        # Display chart
                        st.subheader("📈 Anomaly Score Over Time")
                        
                        if len(results["data"]) > 0:
                            # Create DataFrame for chart
                            df = pd.DataFrame(results["data"])
                            
                            # Create line chart
                            st.line_chart(
                                df.set_index("time_s")["score"],
                                use_container_width=True,
                                height=400
                            )
                            
                            # Add threshold line at 0.5
                            st.caption("Score range: 0.0 (normal) to 1.0 (anomalous). Threshold: 0.5")
                            
                            # Display data table in expander
                            with st.expander("📋 View Raw Data"):
                                st.dataframe(df, use_container_width=True)
                        else:
                            st.warning("No data points to display.")
                        
                        # Detailed results in expander
                        with st.expander("📋 Detailed Results"):
                            st.write(f"**Normal Prompt:** {normal_prompt}")
                            st.write(f"**Anomaly Prompt:** {anomaly_prompt}")
                            st.write(f"**Sampling Rate:** {sampling_rate} fps")
                            st.write("")
                            st.write(f"**Average Anomaly Score:** {metadata.get('average_score', 0):.4f}")
                            st.write(f"**Maximum Anomaly Score:** {metadata.get('max_anomaly_score', 0):.4f} at {metadata.get('max_anomaly_time', 0):.1f}s")
                            st.write(f"**Score Range:** {metadata.get('min_score', 0):.4f} - {metadata.get('max_score', 0):.4f}")
                            st.write(f"**Standard Deviation:** {metadata.get('std_score', 0):.4f}")
                    else:
                        st.error(f"❌ Analysis failed: {results.get('error_message', 'Unknown error')}")
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.exception(e)
                    # Clean up temporary file on error
                    try:
                        if 'tmp_video_path' in locals():
                            os.unlink(tmp_video_path)
                    except:
                        pass


if __name__ == "__main__":
    render_ui_shell()

