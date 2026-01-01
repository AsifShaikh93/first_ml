from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np
import joblib
import os
import tempfile

app = FastAPI()

# UI setup
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# ENV: Model Registry URI
MODEL_URI = os.getenv("MODEL_URI")

if not MODEL_URI:
    raise RuntimeError("MODEL_URI environment variable is not set")

model = None


class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    BMI: float
    Age: int


def download_model(model_uri: str) -> str:
    """
    Download model artifact from Model Registry (HTTP / S3 / MinIO)
    Returns local file path
    """
    print(f"[model] Downloading model from {model_uri}")

    response = requests.get(model_uri)
    response.raise_for_status()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    tmp.write(response.content)
    tmp.close()

    print(f"[model] Model downloaded to {tmp.name}")
    return tmp.name


@app.on_event("startup")
def load_model():
    global model

    model_path = download_model(MODEL_URI)
    model = joblib.load(model_path)

    print("[model] Model loaded successfully")


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
