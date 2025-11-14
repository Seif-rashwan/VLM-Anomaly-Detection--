import streamlit as st
import os

# Set page configuration for a professional look
st.set_page_config(
    page_title="VLM Anomaly Detection",
    layout="wide"
)

# --- Application Layout ---

def render_ui_shell():
    """Renders the minimum required Streamlit UI shell."""
    
    st.title("📹 VLM-Powered Zero-Shot Video Anomaly Detection")
    st.markdown("Upload a video and define the normal and anomalous events using natural language.")

    # --- Sidebar for Inputs ---
    with st.sidebar:
        st.header("1. Upload Video")
        # Widget for video file upload (Phase 2 Deliverable)
        uploaded_file = st.file_uploader(
            "Choose a video file (MP4, MOV, etc.)", 
            type=['mp4', 'mov', 'avi']
        )
        
        st.header("2. Define Conditions")
        # Two text input boxes for the user's natural language prompts (Phase 1 Deliverable)
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

        # The core action button (Phase 1 Deliverable)
        start_analysis_button = st.button("🚨 Start Anomaly Analysis", type="primary")

    # --- Main Content Area (Placeholder for results) ---
    st.header("Analysis Results")
    
    if start_analysis_button:
        # --- PHASE 2/3 INTEGRATION CHECK ---
        # For Step 2, this simply confirms inputs were captured.
        if uploaded_file is not None:
            st.info("Input captured. Ready for Phase 3 Integration.")
            st.write(f"Video: {uploaded_file.name}")
            st.write(f"Normal Prompt: {normal_prompt}")
            st.write(f"Anomaly Prompt: {anomaly_prompt}")
        else:
            st.error("Please upload a video file to start.")

    # Placeholder for the video preview (Phase 2 Deliverable)
    st.subheader("Video Preview")
    if uploaded_file:
        st.video(uploaded_file, format="video/mp4")
    
    # Placeholder for the results visualization (Phase 3 Deliverable)
    st.subheader("Anomaly Score Over Time (Placeholder)")
    st.markdown("*(Line chart visualization will appear here after analysis)*")


if __name__ == "__main__":
    render_ui_shell()
