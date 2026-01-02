# main.py

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np
import joblib
import os
import tempfile
import mlflow
from mlflow.tracking import MlflowClient
import threading
import time

app = FastAPI()

# UI setup
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# ENV: Model Registry URI
MODEL_URI = os.getenv("MODEL_URI")
MODEL_NAME = "diabetes-model"
ALIAS = "Production"

model = None
current_version = None
lock = threading.Lock()




class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    BMI: float
    Age: int





def load_model_if_changed():
    global model, current_version

    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)

    if current_version != mv.version:
        print(f"[model] Loading new model version: v{mv.version}")
        new_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{ALIAS}")

        with lock:
            model = new_model
            current_version = mv.version
            print("[model] Model swapped successfully")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(data: DiabetesInput):
    if model is None:
        return {"error": "Model not loaded"}

    input_data = np.array([[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.BMI,
        data.Age,
    ]])

    prediction = model.predict(input_data)[0]
    return {"diabetes": bool(prediction)}
