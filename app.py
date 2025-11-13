import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = FastAPI()

# Load model
model = load_model("model.h5")

# Emotion classes
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    emotion = classes[np.argmax(preds)]
    confidence = float(np.max(preds))

    return {"emotion": emotion, "confidence": confidence}
