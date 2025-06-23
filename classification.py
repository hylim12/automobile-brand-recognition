# classification.py

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import os
import joblib

# Define paths to the saved PCA features and labels
OUTPUT_DIR = './output'
features_path = os.path.join(OUTPUT_DIR, 'features_pca.npy')
labels_path = os.path.join(OUTPUT_DIR, 'labels.npy')

# Load features and labels
try:
    X = np.load(features_path)
    y = np.load(labels_path)
except FileNotFoundError:
    print("Error: Feature or label file not found. Please run feature_extraction.py first.")
    exit()

# Split into train and test sets (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train all classifiers
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

svm = SVC(probability=True, random_state=42)
svm.fit(X_train, y_train)

mlp = MLPClassifier(random_state=42, max_iter=500)
mlp.fit(X_train, y_train)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Store models in a dict
models = {
    'KNN': knn,
    'SVM': svm,
    'MLP': mlp,
    'RF': rf
}

# Predict on the test set for each classifier
preds = {name: model.predict(X_test) for name, model in models.items()}

# Metrics for each classifier
metrics = {}
for name, y_pred in preds.items():
    metrics[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

class_names_path = os.path.join(OUTPUT_DIR, 'class_names.npy')
if os.path.exists(class_names_path):
    class_names = np.load(class_names_path, allow_pickle=True)
else:
    class_names = None

def classify(features, method='KNN'):
    """
    Classify a single feature vector using the selected model.
    """
    model = models.get(method, knn)
    pred = model.predict([features])[0]
    print(f"Predicted raw: {pred}")
    print(f"Class names: {class_names}")
    # If pred is an integer and class_names is available, map index to name
    if class_names is not None and isinstance(pred, (int, np.integer)):
        brand = class_names[pred]
        print(f"Predicted brand: {brand}")
        return brand
    else:
        print(f"Predicted brand: {pred}")
        return pred

def get_metrics(method='KNN'):
    """
    Return accuracy, precision, recall, f1, report, and confusion matrix for the selected classifier.
    """
    m = metrics.get(method, metrics['KNN'])
    return m['accuracy'], m['precision'], m['recall'], m['f1'], m['report'], m['confusion_matrix']