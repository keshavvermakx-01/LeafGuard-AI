"""
LeafGuard AI - Evaluation Module
Provides functions to load trained model, run evaluation on the test dataset (data/test/),
compute classification metrics (Accuracy, Precision, Recall, F1-Score), and generate confusion matrices.
"""

from pathlib import Path
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

try:
    import tensorflow as tf
except ImportError:
    tf = None

from src.config import (
    TEST_DATA_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    MODEL_SAVE_PATH,
)


def calculate_metrics(y_true, y_pred):
    """
    Calculates key evaluation metrics for crop disease detection model.
    
    Args:
        y_true: True ground-truth labels (1D array or list).
        y_pred: Predicted class labels (1D array or list).
        
    Returns:
        Dictionary containing accuracy, precision, recall, and f1_score.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }
    return metrics


def generate_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Computes confusion matrix for model predictions against ground truth labels.
    
    Args:
        y_true: True ground-truth labels.
        y_pred: Predicted class labels.
        class_names: List of class labels/names.
        
    Returns:
        2D numpy array representing confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    return cm


def print_evaluation_report(y_true, y_pred, class_names=None):
    """
    Prints a detailed classification report including per-class metrics.
    
    Args:
        y_true: True ground-truth labels.
        y_pred: Predicted class labels.
        class_names: List of class labels/names.
    """
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print("\n=== LeafGuard AI Model Classification Report ===")
    print(report)
    return report


def load_test_generator(test_dir=TEST_DATA_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE):
    """
    Creates ImageDataGenerator for the test dataset with shuffle=False.
    
    Args:
        test_dir (Path): Path to test dataset directory.
        target_size (tuple): Image dimensions (height, width).
        batch_size (int): Batch size.
        
    Returns:
        DirectoryIterator: Test dataset generator.
    """
    if tf is None:
        raise ImportError("TensorFlow must be installed to create test dataset generator.")

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return test_generator


def evaluate_model(model_path=MODEL_SAVE_PATH, test_dir=TEST_DATA_DIR):
    """
    Loads trained model and evaluates performance on the test dataset.
    
    Args:
        model_path (Path): Path to trained .h5 model file.
        test_dir (Path): Path to test data directory.
        
    Returns:
        tuple: (metrics_dict, confusion_matrix_array, classification_report_str)
    """
    if tf is None:
        raise ImportError("TensorFlow must be installed to run evaluation.")

    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Trained model file not found at '{model_file}'. "
            "Please train the model using `python -m src.train` before running evaluation."
        )

    print("=== LeafGuard AI Model Evaluation Pipeline ===")
    print(f"Loading trained model from: {model_file}")
    model = tf.keras.models.load_model(str(model_file))

    print(f"Loading test dataset from: {test_dir}")
    test_gen = load_test_generator(test_dir=test_dir)
    
    class_names = list(test_gen.class_indices.keys())
    print(f"Evaluating across {len(class_names)} target classes ({test_gen.samples} test images)...")

    # Run inference on test generator
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes

    # Compute metrics
    metrics = calculate_metrics(y_true, y_pred)
    cm = generate_confusion_matrix(y_true, y_pred, class_names=class_names)
    report = print_evaluation_report(y_true, y_pred, class_names=class_names)

    print("\n--- Summary Performance Metrics ---")
    print(f" Overall Accuracy : {metrics['accuracy'] * 100:.2f}%")
    print(f" Weighted Precision: {metrics['precision'] * 100:.2f}%")
    print(f" Weighted Recall   : {metrics['recall'] * 100:.2f}%")
    print(f" Weighted F1-Score : {metrics['f1_score'] * 100:.2f}%")
    print("===================================\n")

    return metrics, cm, report


if __name__ == "__main__":
    evaluate_model()
