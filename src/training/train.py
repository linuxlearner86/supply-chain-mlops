import pandas as pd
import numpy as np
import os
import joblib
import time
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor


# ----------------------------
# CONFIG
# ----------------------------

DATA_PATH = "data/raw/sales.csv"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "demand_forecast_model.pkl")


# ----------------------------
# LOAD DATA
# ----------------------------

def load_data(path):

    print("Loading FULL dataset...")
    start = time.time()

    df = pd.read_csv(path)

    print(f"Dataset shape: {df.shape}")
    print(f"Load time: {time.time() - start:.2f} seconds")

    return df


# ----------------------------
# PREPROCESS
# ----------------------------

def preprocess(df):

    print("Preprocessing data...")
    start = time.time()

    df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    # faster encoding
    df["item_id"] = df["item_id"].astype("category").cat.codes
    df["store_id"] = df["store_id"].astype("category").cat.codes

    X = df[["item_id", "store_id", "price_base", "year", "month", "day"]]
    y = df["quantity"]

    print(f"Preprocess time: {time.time() - start:.2f} seconds")

    return X, y


# ----------------------------
# TRAIN MODEL
# ----------------------------

def train(X_train, y_train):

    print("Training model on FULL dataset...")
    start = time.time()

    model = HistGradientBoostingRegressor(
        max_iter=100,
        learning_rate=0.1,
        max_depth=10
    )

    model.fit(X_train, y_train)

    print(f"Training time: {time.time() - start:.2f} seconds")

    return model


# ----------------------------
# EVALUATE
# ----------------------------

def evaluate(model, X_test, y_test):

    print("Evaluating model...")

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return mae, rmse


# ----------------------------
# SAVE MODEL
# ----------------------------

def save_model(model):

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print(f"Model saved at {MODEL_PATH}")


# ----------------------------
# MAIN pipeline
# ----------------------------

def main():
    #set experiment name
    mlflow.set_experiment("supply_chain_demand_forcasting")

    with mlflow.start_run():

        print("MLFlow started...")
        
        df = load_data(DATA_PATH)

        X, y = preprocess(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )

        model = train(X_train, y_train)

        evaluate(model, X_test, y_test)

        save_model(model)

if __name__ == "__main__":
    main()
