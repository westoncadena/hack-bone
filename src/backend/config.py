import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = Path("data/models")
DATA_DIR = Path('/data/bs-80k/temp')

# Model configuration
SELECTED_REGIONS = [
    'headANT',
    'chestLANT',
    'chestRANT', 
    'pelvisANT',
    'kneeLANT',
    'kneeRANT'
]

# Image processing configuration
IMAGE_SIZE = 256
CROP_SIZE = 224

# API configuration
HOST = os.getenv('HOST', 'localhost')
PORT = int(os.getenv('PORT', 8000)) 