import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Configurations
IMAGE_FOLDER = "data/captcha_images/"
LABELS_FILE = "data/labels.txt"
CAPTCHA_LENGTH = 6
CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
NUM_CLASSES = len(CHARACTERS)
IMAGE_WIDTH, IMAGE_HEIGHT = 200, 80

char_to_index = {char: i for i, char in enumerate(CHARACTERS)}
index_to_char = np.array(list(CHARACTERS))

def encode_label(text):
    """
    Encode text label to one-hot encoded format
    """
    encoded = np.zeros((CAPTCHA_LENGTH, NUM_CLASSES))
    for i, char in enumerate(text):
        encoded[i, char_to_index[char]] = 1
    return encoded

def load_dataset():
    """
    Load and preprocess the dataset
    Returns: X_train, X_test, y_train, y_test
    """
    images, labels = [], []
    
    # Check if required files/folders exist
    if not os.path.exists(LABELS_FILE):
        raise FileNotFoundError(f"Labels file not found at {LABELS_FILE}")
    if not os.path.exists(IMAGE_FOLDER):
        raise FileNotFoundError(f"Images folder not found at {IMAGE_FOLDER}")
        
    # Read and process data
    with open(LABELS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            try:
                filename, text = line.split(",")
                if len(text) != CAPTCHA_LENGTH:
                    continue
                    
                image_path = os.path.join(IMAGE_FOLDER, filename)
                if not os.path.exists(image_path):
                    continue
                    
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                    
                # Preprocess image
                image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
                images.append(image)
                labels.append(encode_label(text))
                
            except Exception as e:
                print(f"Error processing line '{line}': {str(e)}")
                continue
                
    if not images:
        raise ValueError("No valid images found!")
        
    # Convert to numpy arrays and normalize
    images = np.array(images, dtype=np.float32).reshape(-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1) / 255.0
    labels = np.array(labels, dtype=np.float32)
    
    # Split dataset
    return train_test_split(images, labels, test_size=0.2, random_state=42)

if __name__ == "__main__":
    # Test the dataset loading
    try:
        X_train, X_test, y_train, y_test = load_dataset()
        print(f"Dataset loaded successfully:")
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Input shape: {X_train[0].shape}")
        print(f"Label shape: {y_train[0].shape}")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")