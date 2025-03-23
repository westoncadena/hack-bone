import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import logging

from .config import IMAGE_SIZE, CROP_SIZE, SELECTED_REGIONS, DATA_DIR

logger = logging.getLogger(__name__)

def load_image(image_path: str) -> Image.Image:
    """Load and validate image file"""
    try:
        return Image.open(image_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Error loading image: {str(e)}")

def get_region_paths(filename):
    """
    Get paths to region-specific images corresponding to the uploaded file.
    
    This function should return a dictionary mapping region names to file paths.
    """
    logger.info(f"Looking for region images for file: {filename}")
    
    # Extract the base name without extension
    base_name = Path(filename).stem
    logger.info(f"Base name: {base_name}")
    
    # Initialize the result dictionary
    region_paths = {}
    
    # For each region, check the corresponding directory
    for region in SELECTED_REGIONS:
        # Construct the path to the region-specific directory
        region_dir = Path(f"data/images/temp/{region}")
        logger.info(f"Looking in directory: {region_dir}")
        
        # Check if the directory exists
        if not region_dir.exists():
            logger.error(f"Region directory does not exist: {region_dir}")
            continue
        
        # Construct the expected filename
        region_file = region_dir / f"{base_name}.jpg"
        logger.info(f"Looking for file: {region_file}")
        
        # Check if the file exists
        if region_file.exists():
            region_paths[region] = str(region_file)
            logger.info(f"Found region image for {region}: {region_file}")
        else:
            # Try with other common extensions if jpg doesn't exist
            for ext in ['.png', '.jpeg', '.tif', '.bmp']:
                alt_file = region_dir / f"{base_name}{ext}"
                if alt_file.exists():
                    region_paths[region] = str(alt_file)
                    logger.info(f"Found region image for {region}: {alt_file}")
                    break
            else:
                logger.warning(f"No matching file found for region {region}")
    
    return region_paths

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for model inference"""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0) 