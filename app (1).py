
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Set page config
st.set_page_config(page_title="Sign Language Translator", page_icon="🤟")

st.title("🤟 Sign Language Recognition")
st.write("Upload an image or use your webcam to predict the sign language letter!")

@st.cache_resource
def load_model_once():
    model = load_model('model_new.h5')
    return model

model = load_model_once()

labels = {i: chr(65+i) for i in range(26)}

uploaded_file = st.file_uploader("Choose an image file (JPG/PNG) or use camera input", type=["jpg", "jpeg", "png"])
use_camera = st.checkbox("Use webcam to capture image")

if use_camera:
    picture = st.camera_input("Take a picture")
    if picture:
        file_bytes = np.asarray(bytearray(picture.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
else:
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

if (uploaded_file is not None) or (use_camera and picture):
    with st.spinner('Predicting...'):
        img_resized = cv2.resize(img, (64, 64))
        img_resized = img_resized / 255.0
        img_resized = np.expand_dims(img_resized, axis=0)

        prediction = model.predict(img_resized)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = labels.get(predicted_class, 'Unknown')

        st.image(img, caption=f"Prediction: {predicted_label}", use_column_width=True)
        st.success(f"The predicted sign language letter is: {predicted_label}")

st.sidebar.title("About")
st.sidebar.info("This app uses CNN to recognize American Sign Language (ASL) letters. Created with ❤️ using Streamlit.")
