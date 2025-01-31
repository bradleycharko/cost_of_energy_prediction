# /pre_processing.py

import requests
import pandas as pd
import os
from src.config import EIA_API_KEY, EIA_ENDPOINT, RAW_DATA_PATH, WEATHER_ENDPOINT, WEATHER_API_KEY

def fetch_energy_data(parameters: dict) -> pd.DataFrame:
    """
    Fetch energy price data from an API (e.g., EIA, ERCOT).
    """
    try:
        response = requests.get(EIA_ENDPOINT, params={**parameters, "api_key": EIA_API_KEY})
        response.raise_for_status()
        data = response.json()
        # Convert JSON to DataFrame (this will vary based on API response structure)
        df = pd.DataFrame(data["response"]["data"])
        return df
    except Exception as e:
        print(f"Error fetching energy data: {e}")
        return pd.DataFrame()

def fetch_weather_data(city: str) -> pd.DataFrame:
    """
    Fetch current or historical weather data using an API (e.g., OpenWeatherMap).
    """
    params = {
        "q": city,
        "appid": WEATHER_API_KEY,
        "units": "metric"
    }
    try:
        response = requests.get(WEATHER_ENDPOINT + "weather", params=params)
        response.raise_for_status()
        weather_json = response.json()
        # Convert JSON to DataFrame (simplistic example)
        df = pd.json_normalize(weather_json)
        return df
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return pd.DataFrame()

def save_raw_data(df: pd.DataFrame, filename: str):
    """
    Save raw data to CSV in the data/raw folder.
    """
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    file_path = os.path.join(RAW_DATA_PATH, filename)
    df.to_csv(file_path, index=False)
    print(f"Saved raw data to {file_path}")
