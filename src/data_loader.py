# src/data_loader.py

import pandas as pd
from config import DATA_PATH
from cleaning import basic_cleaning


def load_raw_data() -> pd.DataFrame:
    
    return pd.read_csv(DATA_PATH)


def load_dataset():

    df = load_raw_data()
    df = basic_cleaning(df)

    if "power_consumption" not in df.columns:
        raise ValueError("Target column 'power_consumption' not found in dataset")

    X = df.drop(columns=["power_consumption"])
    y = df["power_consumption"]

    return X, y, df