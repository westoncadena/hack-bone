import streamlit as st
import requests
from pathlib import Path
import json

# Constants
API_URL = "http://localhost:8000/predict"
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg'}

def is_valid_file(filename: str) -> bool:
    """Check if file has allowed extension"""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def main():
    st.set_page_config(
        page_title="Bone Scan Analyzer",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("Bone Scan Analyzer")
    
    # Information section
    with st.expander("About this project", expanded=True):
        st.markdown("""
        ## Bone Metastasis Detection using AI
        
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor 
        incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis 
        nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
        
        ### Model Performance
        - Accuracy: 88.10%
        - Sensitivity: 73.71%
        - Specificity: 96.34%
        - F1 Score: 81.85%
        - AUC: 89.13%
        
        ### How it works
        1. Upload a whole-body bone scan image
        2. Our AI system analyzes specific regions of interest
        3. Get instant results about potential bone metastasis
        """)
    
    # File upload section
    st.header("Upload Bone Scan")
    uploaded_file = st.file_uploader(
        "Drag and drop a whole-body bone scan image",
        type=['jpg', 'jpeg']
    )
    
    if uploaded_file:
        if not is_valid_file(uploaded_file.name):
            st.error("Please upload a JPG image file")
            return
            
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # Process button
        if st.button("Analyze Image"):
            with st.spinner("Analyzing image..."):
                try:
                    files = {"file": uploaded_file}
                    response = requests.post(API_URL, files=files)
                    response.raise_for_status()
                    result = response.json()
                    
                    # Display results
                    st.success("Analysis Complete!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Metastasis Probability",
                            f"{result['probability_positive']:.1%}"
                        )
                    
                    with col2:
                        st.metric(
                            "Normal Probability",
                            f"{result['probability_negative']:.1%}"
                        )
                        
                    with col3:
                        prediction = "Positive" if result['prediction'] > 0.5 else "Negative"
                        st.metric("Final Prediction", prediction)
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"Error analyzing image: {str(e)}")

if __name__ == "__main__":
    main() 