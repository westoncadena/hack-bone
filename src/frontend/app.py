import streamlit as st
from pathlib import Path

# Set page config with dark theme
st.set_page_config(
    page_title="Bone Scan Analyzer",
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

.main-content {
    padding: 2rem;
    background-color: var(--bg-secondary);
    border-radius: 0.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    margin-top: 1rem;
    border: 1px solid var(--border-color);
}

.feature-card {
    background-color: var(--bg-card);
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-top: 1rem;
    border: 1px solid var(--border-color);
    height: 100%;
    min-height: 380px;
    display: flex;
    flex-direction: column;
    color: var(--text-primary);
}

.feature-card h3 {
    color: var(--text-primary);
}

.feature-card p, .feature-card ul {
    color: var(--text-secondary);
}

.feature-icon {
    font-size: 2rem;
    margin-bottom: 1rem;
    color: var(--accent-primary);
}

.cta-button {
    background-color: var(--accent-primary);
    color: white;
    padding: 0.75rem 1.5rem;
    border-radius: 0.5rem;
    text-decoration: none;
    font-weight: 600;
    display: inline-block;
    margin-top: 1.5rem;
    border: none;
    cursor: pointer;
}

.cta-button:hover {
    background-color: var(--accent-secondary);
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

# Custom header
st.markdown("""
<div class="header-container">
    <div>
        <h1 class="header-title">Bone Scan Analyzer</h1>
        <p class="header-subtitle">AI-powered detection of bone metastasis</p>
    </div>
    <div>🔬</div>
</div>
""", unsafe_allow_html=True)

# Hero section
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("""
    # Detect Bone Metastasis with AI
    
    Our advanced AI system analyzes whole-body bone scans to detect potential metastatic lesions with high accuracy.
    
    The Bone Scan Analyzer uses deep learning and computer vision to identify patterns that may indicate the presence of bone metastasis, helping clinicians make more informed decisions.
    """)
    
    # Call to action button
    if st.button("Start Analyzing", type="primary"):
        # Redirect to the analyzer page
        st.switch_page("pages/analyzer.py")


# Features section
st.markdown("## Key Features")

feat_col1, feat_col2, feat_col3 = st.columns(3)

with feat_col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔍</div>
        <h3>High Accuracy</h3>
        <p>Our model achieves:</p>
        <ul>
            <li>88.10% Accuracy</li>
            <li>73.71% Sensitivity</li>
            <li>96.34% Specificity</li>
            <li>81.85% F1 Score</li>
            <li>89.13% AUC</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with feat_col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">⚡</div>
        <h3>Fast Analysis</h3>
        <p>Get results in seconds:</p>
        <ul>
            <li>Upload your scan</li>
            <li>Automatic region detection</li>
            <li>Instant probability scores</li>
            <li>Clear visual indicators</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with feat_col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🧠</div>
        <h3>Advanced Technology</h3>
        <p>Powered by state-of-the-art AI:</p>
        <ul>
            <li>Deep learning feature extraction</li>
            <li>ResNet34 architecture</li>
            <li>Random Forest classification</li>
            <li>Region-specific analysis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# How it works section
st.markdown("## How It Works")

step_col1, step_col2, step_col3 = st.columns(3)

with step_col1:
    st.markdown("### 1. Upload")
    st.markdown("Upload a whole-body bone scan image in JPG format.")

with step_col2:
    st.markdown("### 2. Analyze")
    st.markdown("Our AI system analyzes specific regions of interest in the scan.")

with step_col3:
    st.markdown("### 3. Results")
    st.markdown("Get instant results about potential bone metastasis with probability scores.")

# Call to action
st.markdown("""
<div style="text-align: center; margin-top: 2rem;">
    <h2>Ready to analyze your bone scans?</h2>
</div>
""", unsafe_allow_html=True)

# Call to action button - centered
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Go to Analyzer", type="primary", use_container_width=True):
        # Redirect to the analyzer page
        st.switch_page("pages/analyzer.py")

st.markdown('</div>', unsafe_allow_html=True) 