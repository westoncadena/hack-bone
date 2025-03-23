from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import pickle
import numpy as np
from pathlib import Path
import tempfile
import shutil
import logging

from .models import FeatureExtractor
from .utils import load_image, get_region_paths, preprocess_image
from .config import MODEL_DIR, SELECTED_REGIONS

app = FastAPI(title="Bone Scan Analyzer API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
feature_extractors = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up logging at the top of your file
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_models():
    """Load all required models"""
    try:
        # Load feature extractors
        for region in SELECTED_REGIONS:
            model = FeatureExtractor().to(device)
            model.load_state_dict(
                torch.load(
                    MODEL_DIR / f"resnet34_{region}_best.pth",
                    map_location=device
                )
            )
            model.eval()
            feature_extractors[region] = model
            
        # Load Random Forest classifier
        model_data = torch.load(
            MODEL_DIR / "mSegResRF_SPECT_final.pth",
            map_location=device
        )
        
        # Check if the loaded object is a dictionary
        if isinstance(model_data, dict):
            logger.info("RF classifier loaded as dictionary, extracting model...")
            logger.info(f"Available keys: {list(model_data.keys())}")
            
            # Extract the rf_classifier key specifically
            if 'rf_classifier' in model_data:
                rf_classifier = model_data['rf_classifier']
                logger.info(f"Extracted rf_classifier from dictionary, type: {type(rf_classifier)}")
            else:
                raise RuntimeError(f"Could not find 'rf_classifier' key in dictionary. Available keys: {list(model_data.keys())}")
        else:
            # If it's not a dictionary, use it directly
            rf_classifier = model_data
            logger.info(f"RF classifier loaded directly, type: {type(rf_classifier)}")
            
        # Verify that the classifier has the required method
        if not hasattr(rf_classifier, 'predict_proba'):
            logger.error(f"Loaded object does not have predict_proba method: {type(rf_classifier)}")
            
            # If rf_classifier is itself a dictionary, try to extract the model from it
            if isinstance(rf_classifier, dict):
                logger.info(f"rf_classifier is a dictionary with keys: {list(rf_classifier.keys())}")
                if 'model' in rf_classifier:
                    rf_classifier = rf_classifier['model']
                    logger.info(f"Extracted model from rf_classifier dictionary, type: {type(rf_classifier)}")
                elif 'classifier' in rf_classifier:
                    rf_classifier = rf_classifier['classifier']
                    logger.info(f"Extracted classifier from rf_classifier dictionary, type: {type(rf_classifier)}")
                
                # Check again if the extracted object has predict_proba
                if not hasattr(rf_classifier, 'predict_proba'):
                    raise RuntimeError(f"Extracted object still does not have predict_proba method: {type(rf_classifier)}")
            else:
                raise RuntimeError("Invalid RF classifier: missing predict_proba method")
            
        return rf_classifier
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise RuntimeError(f"Error loading models: {str(e)}")

@app.on_event("startup")
async def startup_event():
    global rf_classifier
    rf_classifier = load_models()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict bone metastasis from whole body scan"""
    logger.info(f"Received prediction request for file: {file.filename}")
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
            logger.info(f"Saved temporary file to: {temp_path}")
            
        # Get region image paths
        logger.info("Getting region image paths...")
        region_paths = get_region_paths(file.filename)
        logger.info(f"Region paths: {region_paths}")
        if not region_paths:
            logger.error("Could not find corresponding region images")
            raise HTTPException(
                status_code=400,
                detail="Could not find corresponding region images"
            )
            
        # Extract features from each region
        logger.info("Extracting features from regions...")
        features = []
        for region in SELECTED_REGIONS:
            logger.info(f"Processing region: {region}")
            if region not in region_paths:
                logger.error(f"Missing region image: {region}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing region image: {region}"
                )
                
            logger.info(f"Loading image from: {region_paths[region]}")
            image = load_image(region_paths[region])
            logger.info(f"Preprocessing image for region: {region}")
            input_tensor = preprocess_image(image).to(device)
            
            logger.info(f"Extracting features for region: {region}")
            with torch.no_grad():
                region_features = feature_extractors[region](input_tensor, return_embedding=True)
                features.append(region_features.cpu().numpy().flatten())
                logger.info(f"Features extracted for region {region}, shape: {region_features.shape}")
                
        # Combine features and predict
        logger.info("Combining features from all regions...")
        combined_features = np.concatenate(features).reshape(1, -1)
        logger.info(f"Combined feature shape: {combined_features.shape}")
        
        # Add error handling and debugging for the prediction step
        logger.info("Making prediction with random forest classifier...")
        try:
            logger.info(f"RF classifier type: {type(rf_classifier)}")
            logger.info(f"RF classifier methods: {dir(rf_classifier)}")
            
            # Check if the model has the expected methods
            if hasattr(rf_classifier, 'predict_proba'):
                logger.info("RF classifier has predict_proba method")
            else:
                logger.error("RF classifier does not have predict_proba method")
                
            # Try to get the expected feature dimensions
            if hasattr(rf_classifier, 'n_features_in_'):
                logger.info(f"Expected feature dimensions: {rf_classifier.n_features_in_}")
                logger.info(f"Actual feature dimensions: {combined_features.shape[1]}")
            
            prediction = rf_classifier.predict_proba(combined_features)[0]
            logger.info(f"Prediction successful: {prediction}")
        except Exception as e:
            import traceback
            logger.error(f"Error during prediction: {str(e)}")
            logger.error(f"Feature shape: {combined_features.shape}")
            logger.error(f"RF classifier type: {type(rf_classifier)}")
            logger.error(f"RF classifier methods: {dir(rf_classifier)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # You can also write to a file for debugging
            with open("debug_log.txt", "w") as f:
                f.write(f"Error during prediction: {str(e)}\n")
                f.write(f"Feature shape: {combined_features.shape}\n")
                f.write(f"RF classifier type: {type(rf_classifier)}\n")
                f.write(f"RF classifier methods: {dir(rf_classifier)}\n")
                f.write(f"Traceback: {traceback.format_exc()}\n")
            
            raise HTTPException(
                status_code=500, 
                detail=f"Error during prediction: {str(e)}"
            )
        
        return {
            "prediction": float(prediction[1]),
            "probability_negative": float(prediction[0]),
            "probability_positive": float(prediction[1])
        }
        
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
        logger.info("Temporary file cleaned up") 