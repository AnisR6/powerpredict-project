# src/data_loader.py

import pandas as pd

from config import DATA_PATH
from cleaning import basic_cleaning, remove_constant_columns


def load_raw_data() -> pd.DataFrame:

    return pd.read_csv(DATA_PATH)


def load_dataset():

    df = load_raw_data()

    # Cleaning
    df = basic_cleaning(df)

    if "power_consumption" not in df.columns:
        raise ValueError(
            "Target column 'power_consumption' not found in dataset"
        )

    X = df.drop(columns=["power_consumption"])
    y = df["power_consumption"]

    # Remove constant features
    X, removed_cols = remove_constant_columns(X)

    if removed_cols:
        print("Removed constant columns:", removed_cols)

    return X, y, df