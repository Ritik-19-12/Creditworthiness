import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model and feature names
model = joblib.load("model/credit_model.pkl")
feature_names = joblib.load("model/columns.pkl")

st.set_page_config(page_title="Creditworthiness Predictor", layout="centered")
st.title("🏦 Creditworthiness Prediction App")
st.markdown("Enter the financial and personal attributes to predict if an individual is creditworthy.")

# Mapping feature names to readable labels
display_names = {
    "Status_Checking_Account": "Status of Checking Account",
    "Duration_Months": "Duration of Credit (Months)",
    "Credit_History": "Credit History",
    "Purpose": "Purpose of Loan",
    "Credit_Amount": "Credit Amount",
    "Savings_Account_Bonds": "Savings Account/Bonds",
    "Employment_Since": "Years of Employment",
    "Installment_Rate": "Installment Rate (%)",
    "Personal_Status_Sex": "Personal Status & Sex",
    "Other_Debtors_Guarantors": "Other Debtors / Guarantors",
    "Present_Residence_Since": "Years at Current Residence",
    "Property": "Property Type",
    "Age": "Age (Years)",
    "Other_Installment_Plans": "Other Installment Plans",
    "Housing": "Housing Type",
    "Existing_Credits_Bank": "Existing Credits at Bank",
    "Job": "Job Type",
    "Liable_People": "Number of Dependents",
    "Telephone": "Has Telephone?",
    "Foreign_Worker": "Is Foreign Worker?",
    "Feature_21": "Other Feature 21",
    "Feature_22": "Other Feature 22",
    "Feature_23": "Other Feature 23",
    "Feature_24": "Other Feature 24"
}

# Feature input ranges based on dataset stats
feature_ranges = {
    'Status_Checking_Account': (1, 4),
    'Duration_Months': (4, 72),
    'Credit_History': (0, 4),
    'Purpose': (2, 184),
    'Credit_Amount': (1, 5),
    'Savings_Account_Bonds': (1, 5),
    'Employment_Since': (1, 4),
    'Installment_Rate': (1, 4),
    'Personal_Status_Sex': (1, 4),
    'Other_Debtors_Guarantors': (19, 75),
    'Present_Residence_Since': (1, 3),
    'Property': (1, 4),
    'Age': (1, 2),
    'Other_Installment_Plans': (1, 2),
    'Housing': (1, 2),
    'Existing_Credits_Bank': (0, 1),
    'Job': (0, 1),
    'Liable_People': (0, 1),
    'Telephone': (0, 1),
    'Foreign_Worker': (0, 1),
    'Feature_21': (0, 1),
    'Feature_22': (0, 1),
    'Feature_23': (0, 1),
    'Feature_24': (0, 1)
}

# Collect user input
user_input = []
for feature in feature_names:
    min_val, max_val = feature_ranges.get(feature, (0, 100))
    label = display_names.get(feature, feature)
    val = st.number_input(f"{label} ({min_val} - {max_val})", min_value=min_val, max_value=max_val, value=min_val, step=1)
    user_input.append(val)

# Predict on button click
if st.button("Predict Creditworthiness"):
    input_df = pd.DataFrame([user_input], columns=feature_names)
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    if prediction == 1:
        st.success(f"✅ Prediction: GOOD CREDIT RISK\nConfidence: {prob[1]*100:.2f}%")
    else:
        st.error(f"❌ Prediction: BAD CREDIT RISK\nConfidence: {prob[0]*100:.2f}%")

st.markdown("---")
st.caption("Trained with cost-sensitive Random Forest. Data Source: UCI German Credit Dataset.")
