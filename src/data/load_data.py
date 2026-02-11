import pandas as pd
import os

def load_data(path: str) -> pd.DataFrame:
    """
    Load supply chain saless data
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"file not found at {path}")
    
    df=pd.read_csv(path)
    print("\n Data loaded Successfully")
    print(f"shape: {df.shape}")

    print("\n Sample data: ")
    print(df.head())


if __name__=="__main__":
    data_path="data/raw/sales.csv"

    df=load_data(data_path)