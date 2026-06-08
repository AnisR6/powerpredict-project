# src/cleaning.py

import numpy as np
import pandas as pd

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Springfield_bars" in df.columns:

        df["Springfield_bars"] = df["Springfield_bars"].astype(float)

        # 6-digit corruption → missing decimal shift
        mask_6 = (
            (df["Springfield_bars"] >= 10000)
            & (df["Springfield_bars"] < 1000000)
        )
        df.loc[mask_6, "Springfield_bars"] /= 100

        # 7-digit corruption
        mask_7 = df["Springfield_bars"] >= 1000000
        df.loc[mask_7, "Springfield_bars"] /= 1000

        # invalid values
        df.loc[df["Springfield_bars"] == 0, "Springfield_bars"] = np.nan

    return df


def remove_constant_columns(df: pd.DataFrame):
    # Remove columns with zero variance.
    constant_cols = [
        col
        for col in df.columns
        if df[col].nunique(dropna=False) <= 1
    ]

    df = df.drop(columns=constant_cols)

    return df, constant_cols