# src/feature_engineering.py
# -------------------------------------------------
# Feature Engineering & Scaling for Customer Churn
# -------------------------------------------------

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# -------------------------------------------------
# Load dataset
# -------------------------------------------------
df = pd.read_csv("data/raw/telecom_churn.csv")

# -------------------------------------------------
# Fix TotalCharges column
# -------------------------------------------------
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# -------------------------------------------------
# Drop customerID (not useful for prediction)
# -------------------------------------------------
df.drop("customerID", axis=1, inplace=True)

# -------------------------------------------------
# Encode target variable
# -------------------------------------------------
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# -------------------------------------------------
# Identify categorical columns
# -------------------------------------------------
categorical_cols = df.select_dtypes(include=["object"]).columns

# -------------------------------------------------
# Label Encoding for categorical columns
# -------------------------------------------------
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# -------------------------------------------------
# Feature Scaling
# -------------------------------------------------
scaler = StandardScaler()
numeric_cols = df.drop("Churn", axis=1).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# -------------------------------------------------
# Save processed dataset
# -------------------------------------------------
df.to_csv("data/processed/cleaned_churn_data.csv", index=False)

# -------------------------------------------------
# SAVE SCALER (🔥 THIS WAS MISSING)
# -------------------------------------------------
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Feature engineering completed successfully!")
print("📁 Cleaned dataset saved in data/processed/")
print("💾 Scaler saved as models/scaler.pkl")
