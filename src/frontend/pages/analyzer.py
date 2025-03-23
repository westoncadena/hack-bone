import streamlit as st
import requests
from pathlib import Path
import json
from PIL import Image
import io
import base64

# Constants
API_URL = "http://localhost:8000/predict"
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg'}
MAX_IMAGE_HEIGHT = 400  # Maximum height for displayed images
REGION_IMAGE_HEIGHT = 200  # Height for region images

def is_valid_file(filename: str) -> bool:
    """Check if file has allowed extension"""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

# Set page config
st.set_page_config(
    page_title="Bone Scan Analyzer - Analysis Tool",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme styling
st.markdown("""
<style>
/* Dark theme colors */
:root {
    --bg-primary: #121212;
    --bg-secondary: #1e1e1e;
    --bg-card: #252525;
    --text-primary: #ffffff;
    --text-secondary: #b0b0b0;
    --accent-primary: #4f6df5;
    --accent-secondary: #3a56d4;
    --border-color: #333333;
}

/* Override Streamlit's default styles for dark theme */
.stApp {
    background-color: var(--bg-primary);
}

.header-container {
    background-color: var(--bg-secondary);
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border: 1px solid var(--border-color);
}

.header-title {
    color: var(--text-primary);
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0;
}

.header-subtitle {
    color: var(--text-secondary);
    font-size: 1.1rem;
    margin-top: 0.5rem;
}

.result-container {
    background-color: var(--bg-card);
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-top: 1rem;
    border: 1px solid var(--border-color);
}

.region-images-container {
    margin-top: 1.5rem;
    padding: 1rem;
    background-color: var(--bg-card);
    border-radius: 0.5rem;
    border: 1px solid var(--border-color);
}

.region-image {
    max-height: 200px;
    width: auto;
    margin: 0 auto;
    display: block;
}

.analysis-container {
    background-color: var(--bg-secondary);
    padding: 2rem;
    border-radius: 0.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    border: 1px solid var(--border-color);
}

/* Override Streamlit text colors */
h1, h2, h3, h4, h5, h6, .stMarkdown {
    color: var(--text-primary) !important;
}

p, li {
    color: var(--text-secondary) !important;
}

/* Make sure buttons are visible in dark mode */
.stButton button {
    background-color: var(--accent-primary);
    color: white;
    border: none;
}

.stButton button:hover {
    background-color: var(--accent-secondary);
}
</style>
""", unsafe_allow_html=True)

# File upload section
st.subheader("Upload Bone Scan")
st.write("Upload a whole-body bone scan image to analyze for potential metastasis.")

uploaded_file = st.file_uploader(
    "Drag and drop a whole-body bone scan image",
    type=['jpg', 'jpeg']
)

if uploaded_file:
    if not is_valid_file(uploaded_file.name):
        st.error("Please upload a JPG image file")
    else:
        # Create two columns for image and results
        image_col, results_col = st.columns([1, 1])
        
        with image_col:
            # Display uploaded image with controlled height
            image = Image.open(uploaded_file)
            
            # Calculate new dimensions to maintain aspect ratio
            width, height = image.size
            new_height = min(height, MAX_IMAGE_HEIGHT)
            new_width = int(width * (new_height / height))
            
            # Resize image
            image = image.resize((new_width, new_height))
            
            # Display image
            st.image(image, caption="Uploaded Image", use_column_width=False)
        
        # Reset file pointer after reading
        uploaded_file.seek(0)
        
        # Process button
        if st.button("Analyze Image", type="primary"):
            with st.spinner("Analyzing image..."):
                try:
                    files = {"file": uploaded_file}
                    response = requests.post(API_URL, files=files)
                    response.raise_for_status()
                    result = response.json()
                    
                    # Display results in the results column
                    with results_col:
                        st.success("Analysis Complete!")
                                                
                        st.metric(
                            "Metastasis Probability",
                            f"{result['probability_positive']:.1%}"
                        )
                        
                        st.metric(
                            "Normal Probability",
                            f"{result['probability_negative']:.1%}"
                        )
                        
                        prediction = "Positive" if result['prediction'] > 0.5 else "Negative"
                        st.metric("Final Prediction", prediction)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display region images if available in the response
                    if 'region_paths' in result and result['region_paths']:
                        st.markdown('<div class="region-images-container">', unsafe_allow_html=True)
                        st.subheader("Region Analysis")
                        st.write("The model analyzed the following regions:")
                        
                        # Sort regions alphabetically
                        regions = result['region_paths']
                        sorted_regions = dict(sorted(regions.items()))
                        num_regions = len(sorted_regions)
                        region_cols = st.columns(min(num_regions, 5))  # Limit to 5 columns max
                        
                        for i, (region, path) in enumerate(sorted_regions.items()):
                            with region_cols[i % len(region_cols)]:
                                try:
                                    # Load and display the region image from the path
                                    region_img = Image.open(path)
                                    
                                    # Resize region image
                                    r_width, r_height = region_img.size
                                    new_r_height = min(r_height, REGION_IMAGE_HEIGHT)
                                    new_r_width = int(r_width * (new_r_height / r_height))
                                    region_img = region_img.resize((new_r_width, new_r_height))
                                    
                                    # Display region image
                                    st.image(region_img, caption=f"Region: {region}", use_column_width=True)
                                except Exception as e:
                                    st.error(f"Could not load region image: {region}\nError: {str(e)}")
                                    st.text(f"Path: {path}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"Error analyzing image: {str(e)}")
else:
    # Show placeholder when no file is uploaded
    st.info("Please upload a bone scan image to begin analysis.")



st.markdown('</div>', unsafe_allow_html=True)