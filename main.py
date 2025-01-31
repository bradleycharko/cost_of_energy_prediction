# main.py

import pandas as pd
from src.data_ingestion import fetch_energy_data, save_raw_data, fetch_weather_data
from src.data_preprocessing import clean_data, handle_outliers
from src.feature_engineering import create_lag_features, add_weather_features, add_rolling_stats
from src.model_training import train_test_models
from src.model_evaluation import evaluate_regression

def main():
    # -------------------------
    # 1. Ingest Data
    # -------------------------
    print("Fetching energy data...")
    energy_df = fetch_energy_data({"frequency": "hourly", "data_type": "price"})
    save_raw_data(energy_df, "energy_data.csv")

    print("Fetching weather data...")
    weather_df = fetch_weather_data("Toronto")  # Example city
    save_raw_data(weather_df, "weather_data.csv")

    # -------------------------
    # 2. Preprocess Data
    # -------------------------
    print("Cleaning energy data...")
    energy_df = clean_data(energy_df)
    energy_df = handle_outliers(energy_df, "price")

    # -------------------------
    # 3. Feature Engineering
    # -------------------------
    print("Creating lag features...")
    energy_df = create_lag_features(energy_df, "price", [1, 24])  # 1-hour and 24-hour lags
    print("Adding rolling stats...")
    energy_df = add_rolling_stats(energy_df, "price", window=24)
    
    # In a real scenario, you'd align timestamps for both datasets 
    # and then do something like:
    # combined_df = add_weather_features(energy_df, weather_df)

    combined_df = energy_df.dropna()  # Drop rows that became NaN due to lag or rolling

    # -------------------------
    # 4. Train Model
    # -------------------------
    print("Training model...")
    model, X_test, y_test = train_test_models(combined_df, "price", model_type="random_forest")

    # -------------------------
    # 5. Evaluate Model
    # -------------------------
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    metrics = evaluate_regression(y_test, y_pred)

if __name__ == "__main__":
    main()
