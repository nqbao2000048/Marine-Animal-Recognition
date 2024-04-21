import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

np.set_printoptions(suppress=True)
# Tải model
model = load_model("VGG16.h5", compile=False)

# Class names for marine animals
class_names = ['Clams', 'Corals', 'Crabs', 'Dolphin', 'Eel', 'Fish', 'Jelly Fish', 'Lobster', 'Nudibranchs',
               'Octopus', 'Otter', 'Penguin', 'Puffers', 'Sea Rays', 'Sea Urchins', 'Seahorse', 'Seal', 'Sharks',
               'Shrimp', 'Squid', 'Starfish', 'Turtle_Tortoise', 'Whale']

# Streamlit interface
st.title("Marine Animal Recognition")

# Upload image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image.", width=250)

    # Convert the image to numpy array
    image = Image.open(uploaded_image).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Preprocess the input
    data[0] = normalized_image_array

    # Predict
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Display prediction and confidence score
    st.write("Predicted Animal:")
    st.title(class_name)
    st.write(f"Confidence Score: {confidence_score}")
