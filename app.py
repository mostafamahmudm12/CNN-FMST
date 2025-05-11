# app.py
import streamlit as st
import requests
import os
from PIL import Image
import io
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_SECRET_KEY")

st.set_page_config(page_title="Fashion MNIST Classifier", layout="wide")

# Title and description
st.title("Fashion MNIST Image Classifier")
st.markdown("""
This application allows you to upload an image of a fashion item, and it will classify it into one of the following categories:
- T-Shirt
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle Boot
""")


# Function to make API request
def classify_image(image_bytes):
    try:
        headers = {"X-API-Key": API_KEY}
        # Use the original filename or a default one
        files = {"file": ("image.png", image_bytes, "image/png")}
        
        response = requests.post(f"{API_URL}/classify", headers=headers, files=files)
        
        if response.status_code != 200:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with API: {e}")
        return None
# Function to preprocess image
def preprocess_image(uploaded_file):
    if uploaded_file is None:
        return None
    
    try:
        # Just return the bytes directly from the uploaded file
        return {"image": Image.open(uploaded_file), "bytes": uploaded_file.getvalue()}
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Main application
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        processed = preprocess_image(uploaded_file)
        if processed:
            st.image(processed["image"], caption="Uploaded Image", use_column_width=True)
            
            if st.button("Classify Image"):
                with st.spinner("Classifying..."):
                    result = classify_image(processed["bytes"])
                    
                    if result:
                        with col2:
                            st.header("Classification Result")
                            st.success(f"Classified as: {result['class_name']}")
                            
                            # Display confidence with progress bar
                            st.write(f"Confidence: {result['confidence']}%")
                            st.progress(result['confidence'] / 100)
                            
                            # Display all class labels with their probabilities
                            st.subheader("Class Information")
                            st.json({
                                "class_index": result['class_index'],
                                "class_name": result['class_name'],
                                "confidence": result['confidence']
                            })

# Add some instructions at the bottom
st.markdown("""
### Instructions
1. Upload an image of a fashion item using the file uploader
2. Click the "Classify Image" button
3. View the classification results on the right side

For best results, use clear images with good lighting and minimal background.
""")

# Add information about the model
st.sidebar.header("About the Model")
st.sidebar.write("""
This application uses a Convolutional Neural Network (CNN) trained on the Fashion MNIST dataset.
The model can recognize 10 different categories of fashion items.

The images are preprocessed to match the training data format (grayscale, 28x28 pixels).
""")

# Add sample images
st.sidebar.header("Sample Images")
st.sidebar.write("You can download these sample images to test the classifier:")

# Create columns for sample images
sample_col1, sample_col2 = st.sidebar.columns(2)

# Load sample images from the assets folder
sample_path = os.path.join("src", "assets")
if os.path.exists(sample_path):
    sample_images = [f for f in os.listdir(sample_path) if f.startswith("test_image_")]
    
    if sample_images:
        for i, img_file in enumerate(sample_images):
            col = sample_col1 if i % 2 == 0 else sample_col2
            img_path = os.path.join(sample_path, img_file)
            img = Image.open(img_path)
            col.image(img, caption=f"Sample {i+1}", use_column_width=True)