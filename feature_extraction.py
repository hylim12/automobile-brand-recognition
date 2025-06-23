# feature_extraction.py

import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
import joblib

# Set the dataset path relative to your project folder
DATASET_PATH = './Manual_Cropped_Logo'
OUTPUT_DIR = './output'
IMG_SIZE = (64, 64)
N_COMPONENTS = 50  # Number of PCA components

def load_images(dataset_path, img_size=(64, 64)):
    X, y = [], []
    class_names = []
    for brand in sorted(os.listdir(dataset_path)):
        brand_path = os.path.join(dataset_path, brand)
        if not os.path.isdir(brand_path):
            continue
        class_names.append(brand)
        for img_name in os.listdir(brand_path):
            img_path = os.path.join(brand_path, img_name)
            print("Trying to load:", img_path)  # Debug print
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print("Failed to load:", img_path)  # Debug print
                continue
            img = cv2.resize(img, img_size)
            X.append(img.flatten())
            y.append(brand)
    return np.array(X), np.array(y), class_names

def extract_features_pca(X, n_components=50):
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    return X_pca, pca

def main():
    print("Loading images...")
    X, y, class_names = load_images(DATASET_PATH, IMG_SIZE)
    print(f"Loaded {len(X)} images from {len(class_names)} classes.")

    if len(X) == 0:
        print("No images loaded. Please check your DATASET_PATH and folder structure.")
        return

    print("Extracting PCA features...")
    X_pca, pca_model = extract_features_pca(X, N_COMPONENTS)
    print("PCA features shape:", X_pca.shape)

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save the results locally
    np.save(os.path.join(OUTPUT_DIR, 'features_pca.npy'), X_pca)
    np.save(os.path.join(OUTPUT_DIR, 'labels.npy'), y)
    joblib.dump(pca_model, os.path.join(OUTPUT_DIR, 'pca_model.joblib'))
    np.save(os.path.join(OUTPUT_DIR, 'class_names.npy'), class_names)

    print("Features and PCA model saved in ./output/")


def extract_features(image):
    """
    Loads the trained PCA model and transforms the input image to PCA features.
    Assumes the image is already preprocessed (grayscale, resized, and flattened).
    """
    import joblib
    import numpy as np
    pca_model = joblib.load('./output/pca_model.joblib')
    # Ensure image is a 1D array
    if len(image.shape) > 1:
        image = image.flatten()
    features = pca_model.transform([image])
    return features[0]

if __name__ == "__main__":
    main()
