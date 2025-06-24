import streamlit as st
from PIL import Image
import numpy as np
import cv2
from feature_extraction import extract_features
from classification import classify, get_metrics
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.title("🚗 Automobile Brand Recognition System")

# Create a sidebar for additional options
with st.sidebar:
    st.header("Settings")
    classifier = st.radio(
        "Choose a classification method:",
        ("KNN", "SVM", "MLP"),
        index=1  # Default to SVM
    )
    svm_kernel = None
    method = classifier
    if classifier == "SVM":
        svm_kernel = st.radio(
            "SVM Kernel:",
            ("Linear", "RBF"),
            index=0
        )
        if svm_kernel == "Linear":
            method = "svm_linear"
        else:
            method = "svm_rbf"
    else:
        method = classifier.lower()
    feature_type = st.radio(
        "Choose feature extraction method:",
        ("PCA", "LDA"),
        index=0  # Default to PCA
    )
    st.markdown("---")
    st.info("Note: Upload a clear image of a car logo for best results")

def preprocess_image(image_path):
    """Preprocess the uploaded image to match training format"""
    try:
        # Read and convert to grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not read the image file")
        
        # Resize to match training size (64x64)
        img = cv2.resize(img, (64, 64))
        
        # Apply some basic image enhancement
        img = cv2.equalizeHist(img)
        
        return img
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def display_metrics(metrics):
    """Display classification metrics in an organized way"""
    st.subheader("Model Performance Metrics")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.2%}")
    with col3:
        st.metric("Recall", f"{metrics['recall']:.2%}")
    with col4:
        st.metric("F1 Score", f"{metrics['f1']:.2%}")
    
    # Classification report
    st.subheader("Classification Report")
    st.text(metrics['report'])
    
    # Confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        metrics['confusion_matrix'],
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=metrics.get('classes', []),
        yticklabels=metrics.get('classes', [])
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# Main file uploader
uploaded_file = st.file_uploader(
    "Upload a car logo image", 
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of a car logo for brand recognition"
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Save temporary file for processing
    temp_file = "temp_upload.jpg"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if st.button("Recognize Brand", type="primary"):
        with st.spinner("Processing image..."):
            try:
                # Preprocess the image
                preprocessed = preprocess_image(temp_file)
                
                if preprocessed is not None:
                    # Display the preprocessed image
                    st.subheader("Preprocessed Image")
                    st.image(preprocessed, caption="Grayscale & Resized (64x64)", width=200)
                    
                    # Extract features (with selected feature type)
                    features = extract_features(preprocessed, feature_type=feature_type)
                    
                    # Classify the image (with selected method and feature type)
                    brand, confidence = classify(features, method=method, feature_type=feature_type)
                    
                    # Display results
                    st.success(f"**Predicted Brand:** {brand}")
                    if confidence is not None:
                        st.info(f"**Confidence:** {confidence:.1%}")
                    
                    # Get and display metrics (with selected method and feature type)
                    metrics = get_metrics(method, feature_type=feature_type)
                    metrics['classes'] = np.load('./output/class_names.npy', allow_pickle=True)
                    display_metrics(metrics)
                    
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file):
                    os.remove(temp_file)

# Add some sample images and instructions
st.markdown("---")
st.subheader("How to Use")
st.markdown("""
1. Upload an image containing a car logo
2. Select a classification method (SVM recommended)
3. Click the "Recognize Brand" button
4. View the recognition results and model metrics

For best results, use clear images of logos on neutral backgrounds.
""")
