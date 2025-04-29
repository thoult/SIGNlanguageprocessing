import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model('best_model.h5')

# Mapping prediction index to sign language letters
labels = {i: chr(65 + i) for i in range(26)}  # 0 -> 'A', 1 -> 'B', ..., 25 -> 'Z'

# Function to process and predict
def predict_image(img):
    img = img.convert('RGB')  # Ensure 3 channels
    img = img.resize((64, 64))  # Resize to match model input
    img = np.array(img) / 255.0  # Normalize
    img = img.reshape(1, 64, 64, 3)  # Reshape to (1, 64, 64, 3)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = labels.get(predicted_class, 'Unknown')
    return predicted_class, predicted_label

# Streamlit UI
st.set_page_config(page_title="Sign Language Translator", layout="centered")

st.title("🤟 Sign Language to Text Translator")
st.markdown("Upload an image or take one using the camera to translate sign language into a letter.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Translate Uploaded Image'):
        with st.spinner("Predicting..."):
            pred_class, pred_label = predict_image(image)
            st.success(f"Predicted Letter: **{pred_label}** (Class: {pred_class})")

# Webcam input
st.markdown("---")
st.header("📸 Or Take a Picture Using Webcam")

camera_image = st.camera_input("Take a photo")

if camera_image is not None:
    image = Image.open(camera_image)
    st.image(image, caption='Captured Image', use_column_width=True)

    if st.button('Translate Camera Image'):
        with st.spinner("Predicting..."):
            pred_class, pred_label = predict_image(image)
            st.success(f"Predicted Letter: **{pred_label}** (Class: {pred_class})")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and TensorFlow")

