# src/data_loader.py

import pandas as pd
from config import DATA_DIR

DATA_PATH = DATA_DIR / "powerpredict.csv.zip"

def load_powerpredict():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["power_consumption"])
    y = df["power_consumption"]

    return X, y, df