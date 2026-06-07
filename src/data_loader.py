# src/data_loader.py

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "powerpredict.csv.zip"

def load_powerpredict():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["power_consumption"])
    y = df["power_consumption"]

    return X, y, df