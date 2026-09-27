"""
LeafGuard AI - Prediction & Inference Module
Loads trained disease detection model and performs inference on leaf images.
(Uses trained model weights - no fake predictions)
"""

from pathlib import Path
import numpy as np

from src.config import MODEL_SAVE_PATH, IMAGE_SIZE
from src.preprocessing import preprocess_single_image

try:
    import tensorflow as tf
except ImportError:
    tf = None

# Default 17 LeafGuard AI v1 class names (in alphabetical order matching Keras flow_from_directory)
CLASS_NAMES_V1 = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


class LeafDiseasePredictor:
    """
    Predictor class for loading a trained LeafGuard AI model and running inference.
    """

    def __init__(self, model_path=MODEL_SAVE_PATH, class_names=CLASS_NAMES_V1):
        """
        Initializes the predictor with model path and class names list/dict.
        
        Args:
            model_path: Path to trained Keras model file (.h5 or SavedModel).
            class_names: List or dict mapping class indices to disease names.
        """
        self.model_path = Path(model_path)
        self.class_names = class_names if class_names is not None else CLASS_NAMES_V1
        self.model = None

    def load_model(self):
        """
        Loads the trained model into memory if model file exists.
        """
        if tf is None:
            raise ImportError("TensorFlow must be installed to load the model.")
            
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model file not found at '{self.model_path}'. "
                "Ensure model is trained and saved before running prediction."
            )
            
        self.model = tf.keras.models.load_model(str(self.model_path))
        print(f"Loaded trained LeafGuard AI model from {self.model_path}")

    def predict(self, image_path_or_file):
        """
        Predicts disease class and confidence score for an input leaf image.
        
        Args:
            image_path_or_file: Image file path or file-like stream object.
            
        Returns:
            Dictionary containing predicted_class_index, predicted_class_name, confidence, and probabilities.
        """
        if self.model is None:
            self.load_model()

        # Preprocess input leaf image
        input_data = preprocess_single_image(image_path_or_file, target_size=IMAGE_SIZE)

        # Run model inference
        predictions = self.model.predict(input_data)
        probabilities = predictions[0]

        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])

        if isinstance(self.class_names, list) and predicted_index < len(self.class_names):
            predicted_label = self.class_names[predicted_index]
        elif isinstance(self.class_names, dict):
            predicted_label = self.class_names.get(predicted_index, f"Class_{predicted_index}")
        else:
            predicted_label = f"Class_{predicted_index}"

        return {
            "predicted_class_index": predicted_index,
            "predicted_class_name": predicted_label,
            "confidence": confidence,
            "all_probabilities": probabilities.tolist()
        }


def predict_disease(image_path_or_file, model_path=MODEL_SAVE_PATH, class_names=CLASS_NAMES_V1):
    """
    Helper function to perform disease prediction on a single image.
    
    Args:
        image_path_or_file: Image file path or file-like object.
        model_path: Path to trained model.
        class_names: Optional class index to label map/list.
        
    Returns:
        Prediction dictionary.
    """
    predictor = LeafDiseasePredictor(model_path=model_path, class_names=class_names)
    return predictor.predict(image_path_or_file)


if __name__ == "__main__":
    print("LeafGuard AI Predictor module initialized.")
