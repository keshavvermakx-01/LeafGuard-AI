"""
LeafGuard AI - Preprocessing Module
Provides modular functions for loading, resizing, and normalizing leaf images.
"""

import numpy as np
from PIL import Image

try:
    import tensorflow as tf
except ImportError:
    tf = None

from src.config import IMAGE_SIZE


def load_image(image_path_or_file):
    """
    Loads an image from a file path or file-like object.
    
    Args:
        image_path_or_file: File path string, Path object, or file buffer.
        
    Returns:
        PIL Image object in RGB mode.
    """
    image = Image.open(image_path_or_file)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def resize_image(image, target_size=IMAGE_SIZE):
    """
    Resizes a PIL Image to target dimensions required by the model.
    
    Args:
        image: PIL Image object.
        target_size: Tuple (height, width).
        
    Returns:
        Resized PIL Image object.
    """
    return image.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)


def normalize_image(image_array):
    """
    Normalizes image pixel values to [0, 1] range.
    
    Args:
        image_array: Numpy array of image pixels (0-255).
        
    Returns:
        Normalized numpy array of float32 values.
    """
    image_array = np.asarray(image_array, dtype=np.float32)
    return image_array / 255.0


def preprocess_single_image(image_path_or_file, target_size=IMAGE_SIZE):
    """
    Full preprocessing pipeline for a single leaf image.
    Loads, resizes, normalizes, and adds batch dimension.
    
    Args:
        image_path_or_file: File path or uploaded file object.
        target_size: Target image dimensions (height, width).
        
    Returns:
        Numpy array with shape (1, height, width, 3) ready for model inference.
    """
    pil_img = load_image(image_path_or_file)
    resized_img = resize_image(pil_img, target_size=target_size)
    img_array = np.array(resized_img)
    normalized_array = normalize_image(img_array)
    # Add batch dimension: (1, H, W, C)
    batch_array = np.expand_dims(normalized_array, axis=0)
    return batch_array


def create_data_generators(train_dir, val_dir, target_size=IMAGE_SIZE, batch_size=32):
    """
    Creates TensorFlow ImageDataGenerator pipelines for training and validation datasets.
    Note: Requires dataset to be present in data directories before execution.
    """
    if tf is None:
        raise ImportError("TensorFlow is required for dataset generator creation.")
        
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, val_generator
