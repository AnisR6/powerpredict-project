# src/feature_engineering.py

import numpy as np
import pandas as pd


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    city_groups = ["Bedrock", "Gotham City", "New New York", "Paperopoli", "Springfield"]

    # 1. Temperature range per city
    for city in city_groups:
        df[f"{city}_temp_range"] = df[f"{city}_t_high"] - df[f"{city}_t_low"]

    # 2. Average temperature across cities
    temp_cols = [f"{city}_t" for city in city_groups]
    df["avg_temperature"] = df[temp_cols].mean(axis=1)

    # 3. Average humidity
    humidity_cols = [f"{city}_humidity" for city in city_groups]
    df["avg_humidity"] = df[humidity_cols].mean(axis=1)

    # 4. Average pressure
    pressure_cols = [f"{city}_bars" for city in city_groups]
    df["avg_pressure"] = df[pressure_cols].mean(axis=1)

    # 5. Total rainfall
    rain_cols = []
    for city in city_groups:
        rain_cols.append(f"{city}_rain_short")
        rain_cols.append(f"{city}_rain_long")

    df["total_rain"] = df[rain_cols].sum(axis=1)

    # 6. Temperature non-linearity
    df["avg_temperature_squared"] = df["avg_temperature"] ** 2

    return df

def add_lag_features(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Adds autoregressive (lag / rolling / momentum) features derived from the
    target itself.

    IMPORTANT: `power_consumption` is a time-ordered series (row order = time
    order, confirmed via strong autocorrelation: lag1 ~0.95, lag2 ~0.84,
    lag24 ~0.31). These features only look backwards (`.shift()`), so they
    are safe for genuine one-step-ahead forecasting.

    However, this ONLY holds if the train/valid split is chronological
    (train = earlier rows, valid = later rows). A random/shuffled split will
    leak future values into the "past" lag features and produce an overly
    optimistic MAE that will not hold up in production. Do not use
    `train_test_split(..., shuffle=True)` (the project default) with this
    function - use a chronological split instead.
    """

    X = X.copy()

    lags = [1, 2, 3, 4, 5, 6, 12, 24]
    for lag in lags:
        X[f"y_lag{lag}"] = y.shift(lag)

    windows = [3, 6, 12, 24]
    for w in windows:
        X[f"y_roll_mean_{w}"] = y.shift(1).rolling(w).mean()
        X[f"y_roll_std_{w}"] = y.shift(1).rolling(w).std()

    X["y_diff1"] = y.shift(1) - y.shift(2)
    X["y_diff2"] = y.shift(2) - y.shift(3)

    X["y_ewm_6"] = y.shift(1).ewm(span=6).mean()
    X["y_ewm_24"] = y.shift(1).ewm(span=24).mean()

    return X