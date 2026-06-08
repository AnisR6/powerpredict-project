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