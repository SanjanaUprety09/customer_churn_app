import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Title
st.title(" Customer Churn Prediction App")
st.write("Fill in the customer details below to predict churn.")

# Load model safely
try:
    with open("customer_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please upload `customer_model.pkl`.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Input form
with st.form("churn_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (in months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=300.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0)

    submitted = st.form_submit_button("Predict")

# Predict only after form submission
if submitted:
    try:
        # Convert categorical to numeric
        gender = 1 if gender == "Male" else 0
        senior_citizen = 1 if senior_citizen == "Yes" else 0
        partner = 1 if partner == "Yes" else 0
        dependents = 1 if dependents == "Yes" else 0

        # Create input dataframe
        input_data = pd.DataFrame({
            "gender": [gender],
            "SeniorCitizen": [senior_citizen],
            "Partner": [partner],
            "Dependents": [dependents],
            "tenure": [tenure],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges]
        })

        # Check for missing values
        if input_data.isnull().values.any():
            st.warning("Please fill out all fields before predicting.")
        else:
            prediction = model.predict(input_data)[0]
            if prediction == 1:
                st.error(" The customer is likely to churn.")
            else:
                st.success(" The customer is not likely to churn.")
    except Exception as e:
        st.error(f"Something went wrong during prediction: {e}")
