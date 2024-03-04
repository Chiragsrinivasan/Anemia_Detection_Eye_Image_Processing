import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

def anemia_detection(img_path):
    model = load_model("anemia_detection_model_eye.h5")

    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    threshold = 0.5
    prediction_classes = 1 if prediction[0][0] > threshold else 0

    return prediction_classes

st.title("Eye-Anemia Detection")

# Specify the path to the input image
input_image_path = ""

prediction = anemia_detection(input_image_path)

if prediction == 0:
    print("No Anemia Detected")
elif prediction == 1:
    print("Anemia Detected")