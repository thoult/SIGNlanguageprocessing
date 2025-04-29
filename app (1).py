import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# ✅ Load the trained model
model = load_model('final_model.h5')

# ✅ Get class names from training if available (A-Z assumed)
# This assumes your folders were A, B, C, ... Z
class_names = [chr(i) for i in range(65, 91)]  # ['A', 'B', ..., 'Z']

# ✅ Predict Function with Preprocessing
def predict_image(img):
    img = img.convert('RGB')                      # Ensure 3 channels
    img = img.resize((64, 64))                    # Resize to model's expected input
    img = np.array(img) / 255.0                   # Normalize pixel values
    img = img.reshape(1, 64, 64, 3)               # Reshape for prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = class_names[predicted_class]
    return predicted_label

# ✅ Streamlit UI
st.set_page_config(page_title="Sign Language Translator", layout="centered")
st.title("🤟 Sign Language to Text Translator")
st.markdown("Upload an image or use your webcam to translate sign language into text.")

# 📁 Upload image
uploaded_file = st.file_uploader("Upload an image of a hand sign", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict from Uploaded Image"):
        with st.spinner("Predicting..."):
            result = predict_image(image)
            st.success(f"✅ Predicted Letter: **{result}**")

# 📷 Camera input
st.markdown("---")
st.header("📸 Or Take a Picture Using Webcam")
camera_image = st.camera_input("Take a photo")

if camera_image is not None:
    image = Image.open(camera_image)
    st.image(image, caption="Captured Image", use_column_width=True)

    if st.button("Predict from Camera Image"):
        with st.spinner("Predicting..."):
            result = predict_image(image)
            st.success(f"✅ Predicted Letter: **{result}**")

st.markdown("---")
st.caption("🧠 Model trained on your dataset — powered by Streamlit & TensorFlow.")


