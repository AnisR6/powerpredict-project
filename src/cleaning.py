# src/cleaning.py

import numpy as np
import pandas as pd

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # fix corrupted pressure scaling
    if "Springfield_bars" in df.columns:
        df["Springfield_bars"] = df["Springfield_bars"].astype(float)

        # fix unrealistic values
        df.loc[df["Springfield_bars"] > 10000, "Springfield_bars"] /= 1000

        # replace invalid zeros
        df.loc[df["Springfield_bars"] == 0, "Springfield_bars"] = np.nan

    return df