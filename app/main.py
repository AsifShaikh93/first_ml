from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import numpy as np
import os
import threading
import time

app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Shared model path (PVC mount)
MODEL_PATH = "/mnt/models/diabetes_model.pkl"
model = None
last_mtime = None


class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    BMI: float
    Age: int


def load_model_if_changed():
    """Load or reload the model if the file changed on disk."""
    global model, last_mtime

    if not os.path.exists(MODEL_PATH):
        print(f"[model] Model file not found at {MODEL_PATH}")
        return

    mtime = os.path.getmtime(MODEL_PATH)
    if last_mtime is None or mtime > last_mtime:
        print(f"[model] Loading model from {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        last_mtime = mtime
        print("[model] Model loaded / reloaded successfully")


def watch_model():
    """Background thread: periodically check for model file changes."""
    while True:
        try:
            load_model_if_changed()
        except Exception as e:
            print(f"[model] Error while reloading model: {e}")
        time.sleep(60)  # check every 60 seconds


@app.on_event("startup")
def startup_event():
    # Initial load
    load_model_if_changed()
    # Start background watcher
    t = threading.Thread(target=watch_model, daemon=True)
    t.start()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(data: DiabetesInput):
    if model is None:
        return {"error": "Model not loaded yet. Try again later."}

    input_data = np.array(
        [[
            data.Pregnancies,
            data.Glucose,
            data.BloodPressure,
            data.BMI,
            data.Age,
        ]]
    )
    prediction = model.predict(input_data)[0]
    return {"diabetes": bool(prediction)}
