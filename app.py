import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Title
st.title("📊 Customer Churn Prediction App")
st.write("Fill in the customer details below to predict churn.")

# Load model and features safely
try:
    with open("customer_model.pkl", "rb") as f:
        data = pickle.load(f)
        model = data["model"]
        model_features = data["features"]
except FileNotFoundError:
    st.error("❌ Model file not found. Please upload `customer_model.pkl`.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Input form
with st.form("churn_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=300.0, value=50.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=500.0)

    submitted = st.form_submit_button("Predict")

# After submission
if submitted:
    try:
        input_dict = {
            "gender": gender,
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        input_df = pd.DataFrame([input_dict])

        # Match training preprocessing
        input_df = pd.get_dummies(input_df)

        # Align columns with model training
        for col in model_features:
            if col not in input_df.columns:
                input_df[col] = 0  # Add missing columns

        input_df = input_df[model_features]  # Ensure correct column order

        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.error("⚠️ The customer is likely to churn.")
        else:
            st.success("✅ The customer is not likely to churn.")

    except Exception as e:
        st.error(f"❌ Something went wrong during prediction: {e}")
