# Bone Scan Analyzer

An AI-powered tool for analyzing whole-body bone scans to detect metastasis.

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create required directories:

```bash
mkdir -p data/images data/models
```

4. Place your trained models in the `models` directory:
- ResNet34 models for each region
- Random forest classifier

5. Copy your bone scan dataset to `data/bs-80k/temp`

## Running the Application

1. Start the backend API:

```bash
uvicorn src.backend.main:app --reload
```

2. Start the Streamlit frontend:
```bash
streamlit run src.frontend.app.py
```

3. Open your browser and navigate to http://localhost:8501

## Project Structure

- `src/backend`: FastAPI backend service
- `src/frontend`: Streamlit web interface
- `models`: Trained model files
- `data`: Dataset directory
- `tests`: Test files

## Model Performance

- Accuracy: 88.10%
- Sensitivity: 73.71%
- Specificity: 96.34%
- F1 Score: 81.85%
- AUC: 89.13%