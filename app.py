import streamlit as st
import joblib

st.title("Cryptocurrency Liquidity Predictor")

# Load trained model
model = joblib.load("liquidity_model.pkl")
st.success("Model loaded successfully")

# User inputs
volume = st.number_input("Enter 24h Volume:", min_value=0.0, format="%.6f")
ma_7 = st.number_input("Enter 7-Day Moving Average:", min_value=0.0, format="%.6f")
volatility = st.number_input("Enter Volatility:", min_value=0.0, format="%.6f")
liquidity_ratio = st.number_input("Enter Liquidity Ratio:", min_value=0.0, format="%.6f")

if st.button("Predict Liquidity"):
    features = [[volume, ma_7, volatility, liquidity_ratio]]
    prediction = model.predict(features)[0]
    st.success(f" Predicted Liquidity: {prediction:.6f}")
