# dashboard/app.py

import streamlit as st
import pandas as pd
import joblib

def load_model():
    return joblib.load("../random_forest_model.pkl")

def main():
    st.title("Energy Market Price Prediction")
    
    model = load_model()
    
    st.subheader("Make a Prediction")
    # In a real scenario, gather input features from the user or an API
    user_input = {
        "price_lag_1": 50.0,
        "price_rolling_mean": 49.5,
        # ...
    }
    
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    
    st.write(f"Predicted Price: {prediction:.2f}")

if __name__ == "__main__":
    main()
