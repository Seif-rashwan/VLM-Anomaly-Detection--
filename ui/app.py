import streamlit as st
import os
import sys
from PIL import Image

# Add parent directory to path to import ml_core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Lazy import to avoid loading PyTorch at module level (prevents Streamlit file watcher warnings)
def get_analyze_function():
    """Lazy import of the VLM analysis function."""
    from ml_core.vlm_test import analyze_image_vlm
    return analyze_image_vlm

# Set page configuration for a professional look
st.set_page_config(
    page_title="VLM Anomaly Detection",
    layout="wide"
)

# --- Application Layout ---

def render_ui_shell():
    """Renders the Streamlit UI for VLM-based image anomaly detection."""
    
    st.title("🔍 VLM-Powered Zero-Shot Image Anomaly Detection")
    st.markdown("Upload an image and define the normal and anomalous conditions using natural language prompts.")

    # --- Sidebar for Inputs ---
    with st.sidebar:
        st.header("1. Upload Image")
        # Widget for image file upload
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['jpg', 'jpeg', 'png', 'bmp', 'gif']
        )
        
        st.header("2. Define Conditions")
        # Two text input boxes for the user's natural language prompts
        normal_prompt = st.text_input(
            "Normal Condition Prompt", 
            value="Dog walking peacefully",
            placeholder="e.g., A person walking normally on the street."
        )
        anomaly_prompt = st.text_input(
            "Anomaly Condition Prompt", 
            value="A heavily damaged car after a crash.",
            placeholder="e.g., A person falling down or a car crash."
        )

        # The core action button
        start_analysis_button = st.button("🚨 Start Analysis", type="primary")

    # --- Main Content Area ---
    
    # Display uploaded image
    if uploaded_file is not None:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, caption=uploaded_file.name, use_container_width=True)
    
    # Analysis Results Section
    if start_analysis_button:
        if uploaded_file is None:
            st.error("⚠️ Please upload an image file to start analysis.")
        elif not normal_prompt.strip() or not anomaly_prompt.strip():
            st.error("⚠️ Please provide both normal and anomaly prompts.")
        else:
            # Show loading spinner
            with st.spinner("🔄 Analyzing image with VLM model... This may take a moment on first run."):
                try:
                    # Reset file pointer
                    uploaded_file.seek(0)
                    image = Image.open(uploaded_file)
                    
                    # Lazy load the analysis function
                    analyze_image_vlm = get_analyze_function()
                    
                    # Run VLM analysis
                    results = analyze_image_vlm(
                        image=image,
                        normal_prompt=normal_prompt,
                        anomaly_prompt=anomaly_prompt
                    )
                    
                    # Display results
                    st.success("✅ Analysis complete!")
                    
                    st.subheader("📊 Similarity Results")
                    
                    # Create columns for better layout
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            label=f"Normal Similarity",
                            value=f"{results['normal']:.4f}",
                            help=f"Similarity to: '{normal_prompt}'"
                        )
                    
                    with col2:
                        st.metric(
                            label=f"Anomaly Similarity",
                            value=f"{results['anomaly']:.4f}",
                            help=f"Similarity to: '{anomaly_prompt}'"
                        )
                    
                    # Determine classification
                    is_normal = results['normal'] > results['anomaly']
                    
                    st.subheader("🎯 Classification Result")
                    
                    if is_normal:
                        st.success(f"✅ **NORMAL** - The image is more similar to the normal condition ({results['normal']:.4f} > {results['anomaly']:.4f})")
                    else:
                        st.error(f"🚨 **ANOMALY** - The image is more similar to the anomaly condition ({results['anomaly']:.4f} > {results['normal']:.4f})")
                    
                    # Detailed results in expander
                    with st.expander("📋 Detailed Results"):
                        st.write(f"**Normal Prompt:** {normal_prompt}")
                        st.write(f"**Normal Similarity Score:** {results['normal']:.4f}")
                        st.write("")
                        st.write(f"**Anomaly Prompt:** {anomaly_prompt}")
                        st.write(f"**Anomaly Similarity Score:** {results['anomaly']:.4f}")
                        st.write("")
                        st.write(f"**Difference:** {abs(results['normal'] - results['anomaly']):.4f}")
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.exception(e)


if __name__ == "__main__":
    render_ui_shell()
