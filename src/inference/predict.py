from fastapi import FastAPI
import joblib
import pandas as pd
import os

# Load model
MODEL_PATH = "models/demand_forecast_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found. Train model first.")

model = joblib.load(MODEL_PATH)
print("Model loaded successfully")

# Create FastAPI app
app = FastAPI(title="Supply Chain Demand Forecast API")

# Home route
@app.get("/")
def home():
    return {"message": "Demand Forecast API is running"}

# Prediction route
@app.post("/predict")
def predict(data: dict):

    # Convert input JSON to DataFrame
    df = pd.DataFrame([data])

    # Convert date properly
    df["date"] = pd.to_datetime(df["date"])

    # Feature engineering
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    # Select features (MUST match training exactly)
    X = df[["item_id", "store_id", "price_base", "year", "month", "day"]]

    # Prediction
    prediction = model.predict(X)[0]

    return {
        "predicted_quantity": float(prediction)
    }
