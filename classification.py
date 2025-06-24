from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import os
import joblib

OUTPUT_DIR = './output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load features and labels
X_train_pca = np.load(os.path.join(OUTPUT_DIR, 'X_train.npy'))
X_test_pca = np.load(os.path.join(OUTPUT_DIR, 'X_test.npy'))
y_train = np.load(os.path.join(OUTPUT_DIR, 'y_train.npy'))
y_test = np.load(os.path.join(OUTPUT_DIR, 'y_test.npy'))
class_names = np.load(os.path.join(OUTPUT_DIR, 'class_names.npy'), allow_pickle=True)
le = joblib.load(os.path.join(OUTPUT_DIR, 'label_encoder.joblib'))

# Load LDA features if available
try:
    X_train_lda = joblib.load(os.path.join(OUTPUT_DIR, 'lda_model.joblib')).transform(
        joblib.load(os.path.join(OUTPUT_DIR, 'scaler.joblib')).transform(X_train_pca))
    X_test_lda = joblib.load(os.path.join(OUTPUT_DIR, 'lda_model.joblib')).transform(
        joblib.load(os.path.join(OUTPUT_DIR, 'scaler.joblib')).transform(X_test_pca))
except Exception:
    X_train_lda = None
    X_test_lda = None

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {}
metrics = {}

# --- k-NN ---
knn_pca = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn_pca.fit(X_train_pca, y_train)
knn_pca_pred = knn_pca.predict(X_test_pca)
knn_pca_cv = cross_val_score(knn_pca, X_train_pca, y_train, cv=cv)
models['knn_pca'] = knn_pca
metrics['knn_pca'] = {
    'accuracy': accuracy_score(y_test, knn_pca_pred),
    'precision': precision_score(y_test, knn_pca_pred, average='weighted', zero_division=0),
    'recall': recall_score(y_test, knn_pca_pred, average='weighted', zero_division=0),
    'f1': f1_score(y_test, knn_pca_pred, average='weighted', zero_division=0),
    'cv_mean': knn_pca_cv.mean(),
    'cv_std': knn_pca_cv.std(),
    'report': classification_report(y_test, knn_pca_pred, target_names=class_names),
    'confusion_matrix': confusion_matrix(y_test, knn_pca_pred)
}
if X_train_lda is not None:
    knn_lda = KNeighborsClassifier(n_neighbors=7, weights='distance')
    knn_lda.fit(X_train_lda, y_train)
    knn_lda_pred = knn_lda.predict(X_test_lda)
    knn_lda_cv = cross_val_score(knn_lda, X_train_lda, y_train, cv=cv)
    models['knn_lda'] = knn_lda
    metrics['knn_lda'] = {
        'accuracy': accuracy_score(y_test, knn_lda_pred),
        'precision': precision_score(y_test, knn_lda_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, knn_lda_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, knn_lda_pred, average='weighted', zero_division=0),
        'cv_mean': knn_lda_cv.mean(),
        'cv_std': knn_lda_cv.std(),
        'report': classification_report(y_test, knn_lda_pred, target_names=class_names),
        'confusion_matrix': confusion_matrix(y_test, knn_lda_pred)
    }

# --- SVM ---
svm_linear_pca = SVC(kernel='linear', C=5.0, random_state=42)
svm_linear_pca.fit(X_train_pca, y_train)
svm_linear_pca_pred = svm_linear_pca.predict(X_test_pca)
svm_linear_pca_cv = cross_val_score(svm_linear_pca, X_train_pca, y_train, cv=cv)
models['svm_linear_pca'] = svm_linear_pca
metrics['svm_linear_pca'] = {
    'accuracy': accuracy_score(y_test, svm_linear_pca_pred),
    'precision': precision_score(y_test, svm_linear_pca_pred, average='weighted', zero_division=0),
    'recall': recall_score(y_test, svm_linear_pca_pred, average='weighted', zero_division=0),
    'f1': f1_score(y_test, svm_linear_pca_pred, average='weighted', zero_division=0),
    'cv_mean': svm_linear_pca_cv.mean(),
    'cv_std': svm_linear_pca_cv.std(),
    'report': classification_report(y_test, svm_linear_pca_pred, target_names=class_names),
    'confusion_matrix': confusion_matrix(y_test, svm_linear_pca_pred)
}
svm_rbf_pca = SVC(kernel='rbf', C=100, gamma='scale', random_state=42)
svm_rbf_pca.fit(X_train_pca, y_train)
svm_rbf_pca_pred = svm_rbf_pca.predict(X_test_pca)
svm_rbf_pca_cv = cross_val_score(svm_rbf_pca, X_train_pca, y_train, cv=cv)
models['svm_rbf_pca'] = svm_rbf_pca
metrics['svm_rbf_pca'] = {
    'accuracy': accuracy_score(y_test, svm_rbf_pca_pred),
    'precision': precision_score(y_test, svm_rbf_pca_pred, average='weighted', zero_division=0),
    'recall': recall_score(y_test, svm_rbf_pca_pred, average='weighted', zero_division=0),
    'f1': f1_score(y_test, svm_rbf_pca_pred, average='weighted', zero_division=0),
    'cv_mean': svm_rbf_pca_cv.mean(),
    'cv_std': svm_rbf_pca_cv.std(),
    'report': classification_report(y_test, svm_rbf_pca_pred, target_names=class_names),
    'confusion_matrix': confusion_matrix(y_test, svm_rbf_pca_pred)
}
if X_train_lda is not None:
    svm_linear_lda = SVC(kernel='linear', C=1.0, random_state=42)
    svm_linear_lda.fit(X_train_lda, y_train)
    svm_linear_lda_pred = svm_linear_lda.predict(X_test_lda)
    svm_linear_lda_cv = cross_val_score(svm_linear_lda, X_train_lda, y_train, cv=cv)
    models['svm_linear_lda'] = svm_linear_lda
    metrics['svm_linear_lda'] = {
        'accuracy': accuracy_score(y_test, svm_linear_lda_pred),
        'precision': precision_score(y_test, svm_linear_lda_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, svm_linear_lda_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, svm_linear_lda_pred, average='weighted', zero_division=0),
        'cv_mean': svm_linear_lda_cv.mean(),
        'cv_std': svm_linear_lda_cv.std(),
        'report': classification_report(y_test, svm_linear_lda_pred, target_names=class_names),
        'confusion_matrix': confusion_matrix(y_test, svm_linear_lda_pred)
    }

# --- MLP ---
mlp_pca = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', alpha=0.01, learning_rate='adaptive', max_iter=500, early_stopping=True, validation_fraction=0.1, random_state=42)
mlp_pca.fit(X_train_pca, y_train)
mlp_pca_pred = mlp_pca.predict(X_test_pca)
mlp_pca_cv = cross_val_score(mlp_pca, X_train_pca, y_train, cv=cv)
models['mlp_pca'] = mlp_pca
metrics['mlp_pca'] = {
    'accuracy': accuracy_score(y_test, mlp_pca_pred),
    'precision': precision_score(y_test, mlp_pca_pred, average='weighted', zero_division=0),
    'recall': recall_score(y_test, mlp_pca_pred, average='weighted', zero_division=0),
    'f1': f1_score(y_test, mlp_pca_pred, average='weighted', zero_division=0),
    'cv_mean': mlp_pca_cv.mean(),
    'cv_std': mlp_pca_cv.std(),
    'report': classification_report(y_test, mlp_pca_pred, target_names=class_names),
    'confusion_matrix': confusion_matrix(y_test, mlp_pca_pred)
}
if X_train_lda is not None:
    mlp_lda = MLPClassifier(hidden_layer_sizes=(20, 10), activation='relu', solver='adam', alpha=0.01, max_iter=300, early_stopping=True, validation_fraction=0.1, random_state=42)
    mlp_lda.fit(X_train_lda, y_train)
    mlp_lda_pred = mlp_lda.predict(X_test_lda)
    mlp_lda_cv = cross_val_score(mlp_lda, X_train_lda, y_train, cv=cv)
    models['mlp_lda'] = mlp_lda
    metrics['mlp_lda'] = {
        'accuracy': accuracy_score(y_test, mlp_lda_pred),
        'precision': precision_score(y_test, mlp_lda_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, mlp_lda_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, mlp_lda_pred, average='weighted', zero_division=0),
        'cv_mean': mlp_lda_cv.mean(),
        'cv_std': mlp_lda_cv.std(),
        'report': classification_report(y_test, mlp_lda_pred, target_names=class_names),
        'confusion_matrix': confusion_matrix(y_test, mlp_lda_pred)
    }

# Save all models
for name, model in models.items():
    joblib.dump(model, os.path.join(OUTPUT_DIR, f'{name}.joblib'))

def classify(features, method='knn', feature_type='pca'):
    key = f'{method.lower()}_{feature_type.lower()}'
    if key not in models:
        raise ValueError(f"Unknown model: {key}. Available: {list(models.keys())}")
    model = models[key]
    pred = model.predict([features])[0]
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba([features])[0]
        confidence = np.max(proba)
    else:
        confidence = None
    brand = class_names[pred]
    return brand, confidence

def get_metrics(method='knn', feature_type='pca'):
    key = f'{method.lower()}_{feature_type.lower()}'
    if key not in metrics:
        raise ValueError(f"Unknown model: {key}. Available: {list(metrics.keys())}")
    return metrics[key]