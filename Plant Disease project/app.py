import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model and class names
model = tf.keras.models.load_model("plant_disease_model.h5")
class_names = np.load("class_names.npy", allow_pickle=True)

IMG_SIZE = 64

# Disease → Plant → Solution mapping
disease_solutions = {

    #  TOMATO
    "Tomato_Tomato_YellowLeaf_Curl_Virus": (
        "Tomato",
        "Control whiteflies, use virus-free seedlings, and remove infected plants."
    ),
    "Tomato_Tomato_mosaic_virus": (
        "Tomato",
        "Remove infected plants, disinfect tools, and avoid handling plants when wet."
    ),
    "Tomato_Target_Spot": (
        "Tomato",
        "Remove infected leaves, improve air circulation, and apply fungicides like chlorothalonil."
    ),
    "Tomato_Spider_mites_Two_spotted_spider_mite": (
        "Tomato",
        "Use insecticidal soap, neem oil, and maintain proper humidity."
    ),
    "Tomato_Septoria_leaf_spot": (
        "Tomato",
        "Remove infected leaves, avoid overhead watering, and apply fungicides."
    ),
    "Tomato_Leaf_Mold": (
        "Tomato",
        "Improve ventilation, reduce humidity, and apply fungicide if necessary."
    ),
    "Tomato_Late_blight": (
        "Tomato",
        "Remove infected plants immediately and apply copper-based fungicides."
    ),
    "Tomato_Early_blight": (
        "Tomato",
        "Use crop rotation, remove infected leaves, and apply fungicide."
    ),
    "Tomato_Bacterial_spot": (
        "Tomato",
        "Use certified seeds, avoid overhead watering, and apply copper sprays."
    ),
    "Tomato_healthy": (
        "Tomato",
        "Plant is healthy. Continue proper irrigation and nutrient management."
    ),

    #  POTATO
    "Potato_Late_blight": (
        "Potato",
        "Remove infected plants and apply fungicides immediately."
    ),
    "Potato_Early_blight": (
        "Potato",
        "Use disease-free seeds, crop rotation, and approved fungicides."
    ),
    "Potato_healthy": (
        "Potato",
        "Plant is healthy. Maintain good soil and watering practices."
    ),

    #  PEPPER
    "Pepper_bell_Bacterial_spot": (
        "Pepper",
        "Apply copper-based fungicides and avoid overhead irrigation."
    ),
    "Pepper_bell_healthy": (
        "Pepper",
        "Plant is healthy. Maintain proper field hygiene."
    )
}


st.set_page_config(page_title="Plant Disease Detection", layout="centered")

st.title(" Plant Disease Detection System")
st.write("Upload a plant leaf image to detect plant type, disease, and solution.")

#  DEFINE uploaded_file FIRST
uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

#  USE uploaded_file AFTER DEFINITION
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    predicted_class = class_names[index]
    confidence = np.max(prediction) * 100

    st.subheader(" Prediction Result")

    if predicted_class in disease_solutions:
        plant, solution = disease_solutions[predicted_class]
        st.success(f" Plant: {plant}")
        st.warning(f" Disease: {predicted_class.replace('_', ' ')}")
        st.info(f" Solution: {solution}")
    else:
        st.warning(f" Disease detected: {predicted_class}")

    st.write(f" Confidence: {confidence:.2f}%")
