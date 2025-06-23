import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Loads an image from the given path, converts it to grayscale, resizes it to 128x128, and returns the processed image as a numpy array.
    Modify this function to include your actual pre-processing steps as needed.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize to 128x128 (change as needed)
    img_resized = cv2.resize(img_gray, (128, 128))
    return img_resized

