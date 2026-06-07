# src/data_loader.py

import pandas as pd
import numpy as np
from config import DATA_PATH

def load_raw_data():
    df = pd.read_csv(DATA_PATH)
    return df

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # fix corrupted pressure scaling (already found)
    if "Springfield_bars" in df.columns:
        df["Springfield_bars"] = df["Springfield_bars"].astype(float)

        df.loc[df["Springfield_bars"] > 10000, "Springfield_bars"] /= 1000

        # replace invalid zeros
        df.loc[df["Springfield_bars"] == 0, "Springfield_bars"] = np.nan

    return df

def load_dataset():
    df = load_raw_data()
    df = basic_cleaning(df)

    X = df.drop(columns=["power_consumption"])
    y = df["power_consumption"]

    return X, y, df