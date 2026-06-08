# src/cleaning.py

import numpy as np
import pandas as pd

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:

    # Basic data cleaning based on EDA findings.
    df = df.copy()

    # Springfield pressure corruption
    if "Springfield_bars" in df.columns:

        df["Springfield_bars"] = df["Springfield_bars"].astype(float)

        # Original corrupted values
        df.loc[df["Springfield_bars"] > 10000, "Springfield_bars"] /= 1000

        # Any remaining pressure below realistic range is considered corrupted and replaced by NaN
        df.loc[df["Springfield_bars"] < 900, "Springfield_bars"] = np.nan

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