import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import keras as ks

# Modell laden
model = ks.models.load_model("model.keras")
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

st.title("Gesichtsemotionserkennung")
st.write("Lade ein Bild eines Gesichts hoch, um die Emotion zu erkennen.")

uploaded_file = st.file_uploader("Bild auswählen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L").resize((48, 48))
    st.image(image, caption='Hochgeladenes Bild (Graustufen)', use_container_width=True)

    # Vorverarbeitung
    img_array = np.array(image).reshape(1, 48, 48, 1) / 255.0

    # Vorhersage
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.write(f"**Erkannte Emotion:** {predicted_class}")
