"""
LeafGuard AI - Dataset Preparation Module
Organizes raw plant leaf image datasets into reproducible train (70%), validation (15%), and test (15%) splits
for the 17 target LeafGuard AI v1 crop disease classes.
"""

import argparse
import os
import random
import shutil
from pathlib import Path
from collections import defaultdict

from src.config import (
    RAW_DATA_DIR,
    TRAIN_DATA_DIR,
    VAL_DATA_DIR,
    TEST_DATA_DIR,
)

# Default PlantVillage raw color dataset source location
DEFAULT_PLANTVILLAGE_COLOR_DIR = RAW_DATA_DIR / "PlantVillage-Dataset" / "raw" / "color"

# Supported image file extensions
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Default fixed random seed for reproducible splits
DEFAULT_SEED = 42

# 17 Target PlantVillage classes for LeafGuard AI v1
TARGET_CLASSES = {
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
}


def get_image_files(class_dir):
    """
    Retrieves all valid image files (JPG, JPEG, PNG) from a class directory.

    Args:
        class_dir (Path): Path to the class directory.

    Returns:
        List[Path]: List of valid image file paths.
    """
    images = []
    for file_path in class_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS:
            images.append(file_path)
    return images


def split_and_copy_dataset(
    source_dir=DEFAULT_PLANTVILLAGE_COLOR_DIR,
    train_dir=TRAIN_DATA_DIR,
    val_dir=VAL_DATA_DIR,
    test_dir=TEST_DATA_DIR,
    target_classes=TARGET_CLASSES,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=DEFAULT_SEED,
    copy_files=True
):
    """
    Splits images for the target 17 LeafGuard v1 classes into train (70%), validation (15%),
    and test (15%) directories while preserving class structure.

    Args:
        source_dir (Path): Source raw dataset directory containing class folders.
        train_dir (Path): Output training directory.
        val_dir (Path): Output validation directory.
        test_dir (Path): Output testing directory.
        target_classes (set): Set of class names to filter and process.
        train_ratio (float): Ratio of images for training (default 0.70).
        val_ratio (float): Ratio of images for validation (default 0.15).
        test_ratio (float): Ratio of images for testing (default 0.15).
        seed (int): Random seed for reproducibility.
        copy_files (bool): If True, copies files; if False, moves them.
    """
    source_dir = Path(source_dir)
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)
    test_dir = Path(test_dir)

    # Validate split ratios sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    if not abs(total_ratio - 1.0) < 1e-5:
        raise ValueError(f"Split ratios must sum to 1.0 (got {total_ratio})")

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory '{source_dir}' does not exist.")

    # Filter class subdirectories to process ONLY the 17 selected target classes
    class_dirs = [
        d for d in source_dir.iterdir()
        if d.is_dir() and (target_classes is None or d.name in target_classes)
    ]
    
    if not class_dirs:
        print(f"No matching target class subdirectories found in '{source_dir}'.")
        return

    # Set fixed random seed for reproducible random shuffling
    random.seed(seed)

    # Tracking statistics for summary report
    stats = {
        "classes_count": len(class_dirs),
        "total_images": 0,
        "train_images": 0,
        "val_images": 0,
        "test_images": 0,
        "per_class": defaultdict(dict)
    }

    print(f"\n--- Preparing LeafGuard AI v1 Dataset from '{source_dir}' ---")
    print(f"Selected Classes: {len(class_dirs)} / 17 target classes found")
    print(f"Splits -> Train: {train_ratio:.0%}, Validation: {val_ratio:.0%}, Test: {test_ratio:.0%}")
    print(f"Random Seed: {seed}\n")

    for class_path in sorted(class_dirs):
        class_name = class_path.name
        images = get_image_files(class_path)
        
        if not images:
            print(f"Warning: No valid images found in class folder '{class_name}'. Skipping.")
            continue

        # Deterministically shuffle image paths
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        # Allocate remaining images to test to ensure exact total match
        n_test = n_total - n_train - n_val

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        # Target class directories
        target_train_class = train_dir / class_name
        target_val_class = val_dir / class_name
        target_test_class = test_dir / class_name

        target_train_class.mkdir(parents=True, exist_ok=True)
        target_val_class.mkdir(parents=True, exist_ok=True)
        target_test_class.mkdir(parents=True, exist_ok=True)

        # File operation (copying preserves raw images in data/raw)
        action_fn = shutil.copy2 if copy_files else shutil.move

        for img in train_imgs:
            action_fn(img, target_train_class / img.name)

        for img in val_imgs:
            action_fn(img, target_val_class / img.name)

        for img in test_imgs:
            action_fn(img, target_test_class / img.name)

        # Update dataset statistics
        stats["total_images"] += n_total
        stats["train_images"] += len(train_imgs)
        stats["val_images"] += len(val_imgs)
        stats["test_images"] += len(test_imgs)

        stats["per_class"][class_name] = {
            "total": n_total,
            "train": len(train_imgs),
            "val": len(val_imgs),
            "test": len(test_imgs),
        }

    # Display dataset summary table
    print_dataset_summary(stats)


def print_dataset_summary(stats):
    """
    Prints a detailed summary report of the dataset split.
    """
    print("=" * 70)
    print("         LEAFGUARD AI v1 DATASET SPLIT SUMMARY (17 CLASSES)         ")
    print("=" * 70)
    print(f" Target Classes Selected    : {stats['classes_count']} / 17")
    print(f" Total Images Processed     : {stats['total_images']}")
    print(f" Training Set Images (70%)  : {stats['train_images']}")
    print(f" Validation Set Images (15%): {stats['val_images']}")
    print(f" Testing Set Images (15%)   : {stats['test_images']}")
    print("-" * 70)
    print(f"{'Class Name':<42} | {'Total':<6} | {'Train':<5} | {'Val':<4} | {'Test':<4}")
    print("-" * 70)
    
    for class_name, counts in sorted(stats["per_class"].items()):
        print(f"{class_name:<42} | {counts['total']:<6} | {counts['train']:<5} | {counts['val']:<4} | {counts['test']:<4}")

    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LeafGuard AI Dataset Preparation Utility")
    parser.add_argument(
        "--source",
        type=str,
        default=str(DEFAULT_PLANTVILLAGE_COLOR_DIR),
        help="Path to source raw dataset directory containing class folders."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Fixed random seed for reproducible splitting (default: 42)."
    )
    
    args = parser.parse_args()
    
    source_path = Path(args.source)
    if source_path.exists() and any(d.is_dir() for d in source_path.iterdir()):
        split_and_copy_dataset(source_dir=source_path, seed=args.seed)
    else:
        print("LeafGuard AI Dataset Preparation Utility initialized.")
        print(f"Source raw directory: {source_path}")
        print("Run `python -m src.prepare_dataset` when ready to split dataset.")
