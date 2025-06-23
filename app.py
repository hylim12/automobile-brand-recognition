import streamlit as st
from PIL import Image
import numpy as np
from preprocessing import preprocess_image
from feature_extraction import extract_features
from classification import classify, get_metrics
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Automobile Brand Recognition System")

classifier = st.radio(
    "Choose a classification method:",
    ("KNN", "SVM", "MLP", "RF")
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    image.save("temp_uploaded_image.jpg")  # Save for OpenCV processing
    if st.button("Process & Recognize"):
        preprocessed = preprocess_image("temp_uploaded_image.jpg")
        features = extract_features(preprocessed)
        result = classify(features, method=classifier)
        st.success(f"Recognition Result: {result}")

        # Display metrics and confusion matrix
        accuracy, precision, recall, f1, report, cm = get_metrics(classifier)
        st.write(f"Accuracy: {accuracy:.2%}")
        st.write(f"Precision Score: {precision:.2%}")
        st.write(f"Recall Score: {recall:.2%}")
        st.write(f"F1 Score: {f1:.2%}")
        st.text("Classification Report:\n" + report)

        st.write("Confusion Matrix:")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", ax=ax)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        st.pyplot(fig)