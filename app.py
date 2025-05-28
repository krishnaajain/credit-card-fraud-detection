# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model, scaler, threshold
model = joblib.load("fraud_detection_model.pkl")
scaler = joblib.load("scaler.pkl")
threshold = joblib.load("optimal_threshold.pkl")

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("💳 Credit Card Fraud Detection")
st.markdown("Advanced machine learning system to detect fraudulent credit card transactions in real-time.")
st.divider()

with st.sidebar:
    st.header("📥 Transaction Details")
    amount = st.number_input("💰 Transaction Amount ($)", min_value=0.0, value=100.0)
    hour = st.slider("⏰ Transaction Hour", 0, 23, 12)
    st.subheader("📊 PCA Features (V1 - V28)")
    v_inputs = [st.slider(f"V{i}", -20.0, 20.0, 0.0) for i in range(1, 29)]

# Predict button
if st.button("🔍 Predict Fraud"):
    # Preprocessing
    scaled = scaler.transform([[amount, hour]])
    amount_scaled = scaled[0][0]
    hour_scaled = scaled[0][1]
    features = v_inputs + [amount_scaled, hour_scaled]
    input_df = pd.DataFrame([features], columns=model.feature_names_in_)

    # Predict
    proba = model.predict_proba(input_df)[0][1]
    is_fraud = proba >= threshold

    st.subheader("📈 Prediction Results")
    st.metric("Fraud Probability", f"{proba*100:.2f}%")

    if is_fraud:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Genuine Transaction")

    # Show feature importance
    st.subheader("🔍 Top Influencing Features")
    feature_importance = pd.DataFrame({
        "Feature": model.feature_names_in_,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).head(5)
    st.bar_chart(feature_importance.set_index("Feature"))
else:
    st.info("Enter details and click **Predict Fraud** to start analysis.")
