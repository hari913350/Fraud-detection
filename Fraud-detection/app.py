import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model & scaler
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to check if it's fraudulent.")

# Input form
with st.form("fraud_form"):
    features = []
    for i in range(1, 29):  # V1 to V28
        features.append(st.number_input(f"V{i}", value=0.0, step=0.01))
    amount = st.number_input("Amount", value=0.0, step=0.01)
    time = st.number_input("Time", value=0.0, step=0.01)
    
    submit = st.form_submit_button("Predict")

if submit:
    input_data = np.array([time] + features + [amount]).reshape(1, -1)
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    prob = model.predict_proba(scaled_data)[0][1]

    if prediction == 1:
        st.error(f"🚨 Fraudulent Transaction! Risk score: {prob:.2f}")
    else:
        st.success(f"✅ Legitimate Transaction. Risk score: {prob:.2f}")
