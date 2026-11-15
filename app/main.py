import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from app.schemas import HeartInput, PredictionOut

app = FastAPI(title="Heart Disease Prediction API")

# Load model once at startup
artifact = joblib.load("model/heart_model.joblib")
model, features = artifact["model"], artifact["features"]

@app.get("/", include_in_schema=False)
def root():
    # Redirect the home page to the Swagger UI
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {"model_type": type(model).__name__, "features": features}

@app.post("/predict", response_model=PredictionOut)
def predict(data: HeartInput):
    # Build a one-row DataFrame in the training column order
    row = pd.DataFrame([data.model_dump()])[features]
    return {"heart_disease": bool(model.predict(row)[0])}