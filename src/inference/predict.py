from fastapi import FastAPI
import joblib
import pandas as pd
import os

MODEL_PATH = "models/demand_forecast_model.pkl"

app = FastAPI(title="Supply Chain Demand Forecast API")

model = None  # global model variable


@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully")
    else:
        print("WARNING: Model file not found. API started without model.")


@app.get("/")
def home():
    return {"message": "Demand Forecast API is running"}


@app.post("/predict")
def predict(data: dict):

    if model is None:
        return {"error": "Model not loaded. Please train and deploy model first."}

    df = pd.DataFrame([data])
    df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    X = df[["item_id", "store_id", "price_base", "year", "month", "day"]]

    prediction = model.predict(X)[0]

    return {
        "predicted_quantity": float(prediction)
    }

