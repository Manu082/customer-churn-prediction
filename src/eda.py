# src/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/raw/telecom_churn.csv")

# Convert TotalCharges to numeric (important fix)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Drop rows with missing values
df.dropna(inplace=True)

# -------------------------------
# 1. Churn Distribution
# -------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x="Churn", data=df)
plt.title("Customer Churn Distribution")
plt.show()

# -------------------------------
# 2. Churn vs Contract Type
# -------------------------------
plt.figure(figsize=(7, 5))
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Churn by Contract Type")
plt.xticks(rotation=20)
plt.show()

# -------------------------------
# 3. Monthly Charges vs Churn
# -------------------------------
plt.figure(figsize=(7, 5))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

# -------------------------------
# 4. Tenure vs Churn
# -------------------------------
plt.figure(figsize=(7, 5))
sns.histplot(data=df, x="tenure", hue="Churn", bins=30, kde=True)
plt.title("Tenure Distribution by Churn")
plt.show()

# -------------------------------
# 5. Correlation Heatmap
# -------------------------------
plt.figure(figsize=(10, 6))
numeric_df = df.select_dtypes(include=["int64", "float64"])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
