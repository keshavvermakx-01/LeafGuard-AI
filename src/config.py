"""
LeafGuard AI - Configuration Settings
Contains configurable hyperparameters, directory paths, and model settings.
"""

from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
TRAIN_DATA_DIR = DATA_DIR / "train"
VAL_DATA_DIR = DATA_DIR / "validation"
TEST_DATA_DIR = DATA_DIR / "test"

# Model Directories & Saved Model Path
MODELS_DIR = BASE_DIR / "models"
MODEL_SAVE_PATH = MODELS_DIR / "leafguard_mobilenetv2.h5"

# Image Preprocessing Hyperparameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Training Hyperparameters
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CLASSES = 17  # 17 target plant disease classes for LeafGuard AI v1

# Transfer Learning Model Architecture
BASE_MODEL_NAME = "MobileNetV2"
