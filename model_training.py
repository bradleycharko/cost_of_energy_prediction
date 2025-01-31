# /model_training.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# For XGBoost, you might do: from xgboost import XGBRegressor
import joblib
from src.config import TRAIN_SIZE, RANDOM_SEED, MODEL_PARAMS

def train_random_forest(X: pd.DataFrame, y: pd.Series):
    """
    Train a Random Forest model on the given features and target.
    """
    params = MODEL_PARAMS["random_forest"]
    rf = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=RANDOM_SEED
    )
    rf.fit(X, y)
    return rf

def train_test_models(df: pd.DataFrame, target_col: str, model_type: str = "random_forest"):
    """
    Generic function to split data, train a specified model, and return it.
    """
    # Simple example: direct column-based splitting
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=TRAIN_SIZE, random_state=RANDOM_SEED
    )

    if model_type == "random_forest":
        model = train_random_forest(X_train, y_train)
    else:
        raise NotImplementedError("Only Random Forest is implemented in this example.")
    
    # Save model for reuse
    joblib.dump(model, f"{model_type}_model.pkl")

    print(f"Trained {model_type} model saved to {model_type}_model.pkl.")
    return model, X_test, y_test
