# app.py
# -------------------------------------------------
# Customer Churn Prediction - Streamlit Application
# Includes: Prediction + Segment-wise EDA
# -------------------------------------------------

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load model artifacts
# -----------------------------
model = joblib.load("models/best_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")
scaler = joblib.load("models/scaler.pkl")

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.write("Predict whether a customer is likely to churn and analyze churn patterns.")

st.divider()

# =================================================
# PREDICTION SECTION
# =================================================
st.subheader("🧾 Enter Customer Details")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
tech_support = st.selectbox("Tech Support", ["Yes", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

# -----------------------------
# Build feature row
# -----------------------------
input_dict = dict.fromkeys(feature_names, 0)

input_dict["tenure"] = tenure
input_dict["MonthlyCharges"] = monthly_charges
input_dict["TotalCharges"] = total_charges
input_dict["Contract"] = ["Month-to-month", "One year", "Two year"].index(contract)
input_dict["PaymentMethod"] = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
].index(payment)
input_dict["InternetService"] = ["DSL", "Fiber optic", "No"].index(internet)
input_dict["TechSupport"] = 1 if tech_support == "Yes" else 0
input_dict["OnlineSecurity"] = 1 if online_security == "Yes" else 0
input_dict["PaperlessBilling"] = 1 if paperless == "Yes" else 0

input_df = pd.DataFrame([input_dict])
input_df_scaled = scaler.transform(input_df)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Churn"):
    prediction = model.predict(input_df_scaled)[0]
    probability = model.predict_proba(input_df_scaled)[0][1]

    st.subheader("📊 Prediction Result")

    st.progress(probability)
    st.write(f"**Churn Probability:** `{probability:.2f}`")

    if probability < 0.4:
        st.success("🟢 Low Risk of Churn")
    elif probability < 0.7:
        st.warning("🟡 Medium Risk of Churn")
    else:
        st.error("🔴 High Risk of Churn")

    if prediction == 1:
        st.error("⚠ Model Prediction: Customer is likely to churn")
    else:
        st.success("✅ Model Prediction: Customer is likely to stay")

# =================================================
# EDA SECTION (SEGMENT-WISE)
# =================================================
st.divider()
st.subheader("📈 Exploratory Data Analysis (Segment-wise)")

df = pd.read_csv("data/raw/telecom_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# -----------------------------
# EDA Filters
# -----------------------------
st.markdown("### 🔎 Filter Customer Segment")

col1, col2 = st.columns(2)

with col1:
    eda_contract = st.selectbox(
        "Contract Type (EDA)",
        ["All"] + sorted(df["Contract"].unique().tolist())
    )

with col2:
    eda_payment = st.selectbox(
        "Payment Method (EDA)",
        ["All"] + sorted(df["PaymentMethod"].unique().tolist())
    )

# Apply filters
filtered_df = df.copy()

if eda_contract != "All":
    filtered_df = filtered_df[filtered_df["Contract"] == eda_contract]

if eda_payment != "All":
    filtered_df = filtered_df[filtered_df["PaymentMethod"] == eda_payment]

# -----------------------------
# Charts
# -----------------------------
if filtered_df.empty:
    st.warning("No data available for the selected segment.")
else:
    col1, col2 = st.columns(2)

    # Churn Distribution
    with col1:
        st.markdown("**Churn Distribution**")
        fig1, ax1 = plt.subplots()
        sns.countplot(x="Churn", data=filtered_df, ax=ax1)
        st.pyplot(fig1)

    # Monthly Charges vs Churn
    with col2:
        st.markdown("**Monthly Charges vs Churn**")
        fig2, ax2 = plt.subplots()
        sns.boxplot(x="Churn", y="MonthlyCharges", data=filtered_df, ax=ax2)
        st.pyplot(fig2)

    # Tenure Distribution
    st.markdown("**Tenure Distribution by Churn**")
    fig3, ax3 = plt.subplots()
    sns.histplot(
        data=filtered_df,
        x="tenure",
        hue="Churn",
        bins=30,
        kde=True,
        ax=ax3
    )
    st.pyplot(fig3)

    churn_rate = (
        filtered_df["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
    )

    st.info(
        f"📌 **Segment Insight** → "
        f"Contract: **{eda_contract}**, "
        f"Payment: **{eda_payment}**, "
        f"Churn Rate: **{churn_rate:.2f}%**"
    )

# =================================================
# FOOTER
# =================================================
st.divider()
st.caption("Internship Project | Customer Churn Prediction | Codec Technologies")
