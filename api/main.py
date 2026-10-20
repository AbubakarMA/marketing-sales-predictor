from fastapi import FastAPI
from pydantic import BaseModel, Field
from pathlib import Path
from functools import lru_cache

import pandas as pd
import joblib


# Create the FastAPI application
app = FastAPI(
    title="Marketing Sales Predictor API",
    description="API for predicting sales from TV advertising budgets",
    version="1.0.0",
)


# Locate the project root and saved model
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "marketing_model.joblib"


# Define the expected prediction request
class PredictionRequest(BaseModel):
    tv_budget: float = Field(
        ...,
        ge=0,
        le=300,
        description="TV advertising budget in thousands of dollars",
    )


# Load the model once and reuse it
@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    return model


# Health endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "marketing-sales-predictor-api",
    }


# Prediction endpoint
@app.post("/predict")
def predict_sales(request: PredictionRequest):
    # 1. Load the trained model
    model = load_model()

    # 2. Convert the input into a DataFrame
    input_data = pd.DataFrame({
        "TV": [request.tv_budget]
    })

    # 3. Generate the prediction
    prediction = model.predict(input_data)[0]

    # 4. Return a JSON-compatible response
    return {
        "predicted_sales": round(float(prediction), 4),
        "unit": "thousand_units",
    }
