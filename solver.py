# solver.py
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

class CaptchaSolver:
    def __init__(self, model_path: str, image_width: int = 200, image_height: int = 80):
        """
        Initialize the CAPTCHA solver
        
        Args:
            model_path: Path to the trained model
            image_width: Width of input images
            image_height: Height of input images
        """
        self.image_width = image_width
        self.image_height = image_height
        self.characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        
        # Load model
        try:
            self.model = tf.keras.models.load_model(model_path)
        except Exception as e:
            raise Exception(f"Error loading model from {model_path}: {str(e)}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for prediction
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Preprocessed image tensor
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Resize
            image = cv2.resize(image, (self.image_width, self.image_height))
            
            # Normalize and reshape
            image = image.astype(np.float32) / 255.0
            image = np.expand_dims(image, axis=-1)
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            raise Exception(f"Error preprocessing image: {str(e)}")

    def predict(self, image: np.ndarray) -> str:
        """
        Predict CAPTCHA text from image
        
        Args:
            image: Input image
            
        Returns:
            Predicted CAPTCHA text
        """
        try:
            # Preprocess image
            preprocessed_image = self.preprocess_image(image)
            
            # Make prediction
            predictions = self.model.predict(preprocessed_image, verbose=0)
            
            # Decode prediction
            predicted_text = "".join(
                self.characters[np.argmax(predictions[0, i])]
                for i in range(predictions.shape[1])
            )
            
            return predicted_text
            
        except Exception as e:
            raise Exception(f"Error making prediction: {str(e)}")

def test_solver(model_path: str, test_image_path: str):
    """
    Test the CAPTCHA solver
    """
    try:
        # Initialize solver
        solver = CaptchaSolver(model_path)
        
        # Load and predict test image
        image = cv2.imread(test_image_path)
        if image is None:
            raise ValueError(f"Could not load image from {test_image_path}")
            
        predicted_text = solver.predict(image)
        print(f"Predicted CAPTCHA text: {predicted_text}")
        
    except Exception as e:
        print(f"Error testing solver: {str(e)}")

if __name__ == "__main__":
    # Example usage
    model_path = "models/final_model.keras"
    test_image_path = "data\captcha_images\ZMW8DM.png"
    
    if Path(model_path).exists() and Path(test_image_path).exists():
        test_solver(model_path, test_image_path)
    else:
        print("Please ensure model and test image files exist")