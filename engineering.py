# /engineering.py

import pandas as pd

def create_lag_features(df: pd.DataFrame, column: str, lags: list = [1, 2, 3]) -> pd.DataFrame:
    """
    Create lag features for time-series modeling.
    """
    for lag in lags:
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)
    return df

def add_weather_features(df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Example of merging weather data (temperature, humidity) 
    into main DataFrame based on date/time or location.
    """
    # Assume both dfs have a 'timestamp' column to merge on
    merged_df = pd.merge(df, weather_df, on='timestamp', how='left')
    return merged_df

def add_rolling_stats(df: pd.DataFrame, column: str, window: int = 7) -> pd.DataFrame:
    """
    Adds rolling mean, rolling std as features.
    """
    df[f"{column}_rolling_mean"] = df[column].rolling(window=window).mean()
    df[f"{column}_rolling_std"] = df[column].rolling(window=window).std()
    return df