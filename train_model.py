# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load dataset
df = pd.read_csv('data/german.data-numeric', sep=r'\s+', header=None)
df.columns = [
    "Status_Checking_Account", "Duration_Months", "Credit_History", "Purpose",
    "Credit_Amount", "Savings_Account_Bonds", "Employment_Since", "Installment_Rate",
    "Personal_Status_Sex", "Other_Debtors_Guarantors", "Present_Residence_Since", "Property",
    "Age", "Other_Installment_Plans", "Housing", "Existing_Credits_Bank", "Job",
    "Liable_People", "Telephone", "Foreign_Worker",
    "Feature_21", "Feature_22", "Feature_23", "Feature_24", "Target"
]

df['Target'] = df['Target'].map({1: 1, 2: 0})

# Train/test split
X = df.drop('Target', axis=1)
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest with class weights from cost matrix
rf = RandomForestClassifier(class_weight={0: 5, 1: 1}, n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Save model and feature names
os.makedirs('model', exist_ok=True)
joblib.dump(rf, 'model/credit_model.pkl')
joblib.dump(list(X.columns), 'model/columns.pkl')

print("✅ Model trained and saved successfully!")
