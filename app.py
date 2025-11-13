import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Real-Time Emotion Detector", layout="wide")

# Load Model
model = load_model("model/mobilenet_emotion_model.h5")

# Emotion Classes
classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

st.title("😃 Real-Time Emotion Detection App")

run = st.checkbox("Start Webcam")

FRAME_WINDOW = st.image([])

# Initialize Webcam
camera = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

while run:
    ret, frame = camera.read()
    if not ret:
        st.write("Camera not available.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        preds = model.predict(face, verbose=0)
        label = classes[np.argmax(preds)]
        conf = np.max(preds) * 100

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"{label} ({conf:.1f}%)", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    FRAME_WINDOW.image(frame, channels='BGR')

camera.release()
