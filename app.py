import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the model
model = load_model('maize_disease_model.h5')
classes = ['Blight', 'Gray spot', 'Rust', 'Healthy']

# Function to predict class
def predict_disease(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Resize to model input size
    img_array = np.expand_dims(img_to_array(img) / 255.0, axis=0)  # Normalize
    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return predicted_class, confidence

# Streamlit UI
st.title("Maize Disease Prediction")
st.write("Upload an image of maize leaf to predict the disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Save uploaded image to a temporary file
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image("temp_image.jpg", caption="Uploaded Image", use_column_width=True)

    # Make prediction
    predicted_class, confidence = predict_disease("temp_image.jpg")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
