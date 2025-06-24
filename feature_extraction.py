import os
import cv2
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Set the dataset path relative to your project folder
DATASET_PATH = './Manual_Cropped_Logo'
OUTPUT_DIR = './output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMG_SIZE = (64, 64)

# --- Preprocessing ---
def preprocess_logo_image(image_path, size):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        if len(resized.shape) == 3:
            lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
            l_enhanced = clahe.apply(l)
            lab_enhanced = cv2.merge((l_enhanced, a, b))
            enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
            enhanced = clahe.apply(resized)
        if len(enhanced.shape) == 3:
            gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
        else:
            gray = enhanced
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        normalized = blurred.astype(np.float32) / 255.0
        return normalized
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def simple_augment(image):
    try:
        h, w = image.shape
        angle = np.random.uniform(-5, 5)
        center = (w//2, h//2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        brightness = np.random.uniform(0.85, 1.15)
        adjusted = np.clip(rotated * brightness, 0, 1)
        scale = np.random.uniform(0.95, 1.05)
        new_h, new_w = int(h * scale), int(w * scale)
        scaled = cv2.resize(adjusted, (new_w, new_h))
        if new_h >= h and new_w >= w:
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            final = scaled[start_y:start_y+h, start_x:start_x+w]
        else:
            pad_y = max(0, (h - new_h) // 2)
            pad_x = max(0, (w - new_w) // 2)
            final = np.pad(scaled, ((pad_y, h-new_h-pad_y), (pad_x, w-new_w-pad_x)), mode='edge')
            if final.shape != (h, w):
                final = cv2.resize(final, (w, h))
        return final.astype(np.float32)
    except:
        return None

def balance_dataset_with_augmentation(images, labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    max_count = max(counts)
    target = min(max_count + 15, 160)
    balanced_images = []
    balanced_labels = []
    for label in unique_labels:
        brand_mask = labels == label
        brand_images = images[brand_mask]
        current_count = len(brand_images)
        for img in brand_images:
            balanced_images.append(img)
            balanced_labels.append(label)
        if current_count < target:
            needed = target - current_count
            for i in range(needed):
                base_img = brand_images[i % current_count]
                augmented = simple_augment(base_img)
                if augmented is not None:
                    balanced_images.append(augmented)
                    balanced_labels.append(label)
    return np.array(balanced_images), np.array(balanced_labels)

def load_automobile_logos(dataset_path, img_size, brand_labels):
    images = []
    image_labels = []
    for brand, label in brand_labels.items():
        brand_path = os.path.join(dataset_path, brand)
        if not os.path.exists(brand_path):
            continue
        image_files = [f for f in os.listdir(brand_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        for img_file in image_files:
            full_path = os.path.join(brand_path, img_file)
            processed = preprocess_logo_image(full_path, img_size)
            if processed is not None:
                images.append(processed)
                image_labels.append(label)
    return np.array(images), np.array(image_labels)

def extract_features_pca_lda(X_train, X_test, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    pca_full = PCA(random_state=42)
    pca_full.fit(X_train_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cumvar >= 0.92) + 1
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    pca.fit(X_train_scaled)
    X_train_pca = pca.transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    n_classes = len(np.unique(y_train))
    lda_components = n_classes - 1
    lda = LinearDiscriminantAnalysis(n_components=lda_components, solver='svd')
    lda.fit(X_train_scaled, y_train)
    X_train_lda = lda.transform(X_train_scaled)
    X_test_lda = lda.transform(X_test_scaled)
    return {
        'scaler': scaler,
        'pca': pca,
        'lda': lda,
        'X_train_pca': X_train_pca,
        'X_test_pca': X_test_pca,
        'X_train_lda': X_train_lda,
        'X_test_lda': X_test_lda,
        'n_pca_components': n_components
    }

def main():
    # Brand label mapping
    brand_labels = {brand: idx for idx, brand in enumerate(sorted(os.listdir(DATASET_PATH)))}
    images, labels = load_automobile_logos(DATASET_PATH, IMG_SIZE, brand_labels)
    if images is None or labels is None:
        print("No images loaded!")
        return
    images, labels = balance_dataset_with_augmentation(images, labels)
    num_pixels = IMG_SIZE[0] * IMG_SIZE[1]
    X_flat = images.reshape(len(images), num_pixels)
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, labels, test_size=0.4, random_state=42, stratify=labels)
    features = extract_features_pca_lda(X_train, X_test, y_train)
    # Save all models and splits
    np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), features['X_train_pca'])
    np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'), features['X_test_pca'])
    np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_test)
    np.save(os.path.join(OUTPUT_DIR, 'class_names.npy'), np.array(list(brand_labels.keys())))
    joblib.dump(features['scaler'], os.path.join(OUTPUT_DIR, 'scaler.joblib'))
    joblib.dump(features['pca'], os.path.join(OUTPUT_DIR, 'pca_model.joblib'))
    joblib.dump(features['lda'], os.path.join(OUTPUT_DIR, 'lda_model.joblib'))
    le = LabelEncoder().fit(labels)
    joblib.dump(le, os.path.join(OUTPUT_DIR, 'label_encoder.joblib'))
    print("All data and models saved to ./output.")

def extract_features(image, feature_type='PCA'):
    # Load scaler and model
    scaler = joblib.load('./output/scaler.joblib')
    if len(image.shape) > 1:
        image = image.flatten()
    image = image.astype(np.float32)
    image = image.reshape(1, -1)
    image_scaled = scaler.transform(image)
    if feature_type == 'PCA':
        pca = joblib.load('./output/pca_model.joblib')
        features = pca.transform(image_scaled)
    elif feature_type == 'LDA':
        lda = joblib.load('./output/lda_model.joblib')
        features = lda.transform(image_scaled)
    else:
        raise ValueError('Unknown feature_type. Use "PCA" or "LDA".')
    return features[0]

if __name__ == "__main__":
    main()