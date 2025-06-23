# Automobile Brand Recognition System

This project is a web application for recognizing automobile brands from images using machine learning. It supports multiple classifiers (KNN, SVM, MLP, RF) and provides performance metrics and a confusion matrix for each method.

## Features
- Upload an image of a car logo to recognize its brand
- Choose between KNN, SVM, MLP, and Random Forest classifiers
- View accuracy, precision, recall, F1 score, classification report, and confusion matrix
- User-friendly web interface built with Streamlit

## Setup Instructions

### 1. Clone the Repository
```
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create and Activate a Virtual Environment (Recommended)
```
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```
If you don't have a `requirements.txt`, install manually:
```
pip install streamlit pillow numpy opencv-python scikit-learn joblib matplotlib seaborn pandas
```

### 4. Prepare the Dataset
- Unzip your dataset (e.g., `Manual_Cropped_Logo.zip`) in the project folder.
- Ensure the folder structure is:
  ```
  Project/
    Manual_Cropped_Logo/
      Honda/
      Mazda/
      Perodua/
      Toyota/
  ```

### 5. Extract Features and Train Models
Run the feature extraction script to generate features and models:
```
python feature_extraction.py
```

### 6. Run the Web App
```
streamlit run app.py
```
- The app will open in your browser at `http://localhost:8501`.
- Upload an image and select a classifier to see the recognition result and metrics.

## Troubleshooting
- **ModuleNotFoundError**: Make sure all dependencies are installed in your active environment.
- **PCA feature mismatch**: Ensure your preprocessing resizes images to the same size as during training (default: 64x64).
- **Wrong predictions**: Check that your dataset is correctly structured and that you re-ran `feature_extraction.py` after any changes.
- **Virtual environment issues**: Always activate your virtual environment before running scripts.

## Project Structure
```
app.py                  # Streamlit web app
classification.py       # Classifier training and prediction
feature_extraction.py   # Feature extraction and PCA
preprocessing.py        # Image preprocessing
Manual_Cropped_Logo/    # Dataset (unzip here)
output/                 # Generated features, models, and class names
requirements.txt        # Python dependencies
README.md               # This file
```

## License
This project is for educational purposes. Please cite appropriately if used in academic work.

---
If you have any issues, please open an issue on GitHub or contact the maintainer. 