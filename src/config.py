# src/config.py

import os

# ----------------------
# API Keys & Endpoints
# ----------------------
EIA_API_KEY = os.getenv("EIA_API_KEY", "YOUR_EIA_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "YOUR_WEATHER_API_KEY")
EIA_ENDPOINT = "https://api.eia.gov/v2/electricity-prices"
WEATHER_ENDPOINT = "https://api.openweathermap.org/data/2.5/"

# ----------------------
# File Paths
# ----------------------
RAW_DATA_PATH = os.path.join("data", "raw")
PROCESSED_DATA_PATH = os.path.join("data", "processed")

# ----------------------
# Modeling
# ----------------------
TRAIN_SIZE = 0.8
RANDOM_SEED = 42
MODEL_PARAMS = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10
    },
    "xgboost": {
        "n_estimators": 100,
        "learning_rate": 0.01
    }
}