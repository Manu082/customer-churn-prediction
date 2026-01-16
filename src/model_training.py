# src/model_training.py
# -------------------------------------------------
# Customer Churn Prediction - Model Training Script
# -------------------------------------------------

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# -------------------------------------------------
# Load processed dataset
# -------------------------------------------------
df = pd.read_csv("data/processed/cleaned_churn_data.csv")

# -------------------------------------------------
# Split features and target
# -------------------------------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

# -------------------------------------------------
# Save feature names (VERY IMPORTANT for Streamlit)
# -------------------------------------------------
feature_names = X.columns.tolist()

# -------------------------------------------------
# Train-test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------------------------
# 1. Logistic Regression (Baseline Model)
# -------------------------------------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

lr_preds = lr_model.predict(X_test)
lr_probs = lr_model.predict_proba(X_test)[:, 1]

lr_accuracy = accuracy_score(y_test, lr_preds)
lr_recall = recall_score(y_test, lr_preds)
lr_roc = roc_auc_score(y_test, lr_probs)

print("\n================ Logistic Regression Results ================")
print("Accuracy :", lr_accuracy)
print("Recall   :", lr_recall)
print("ROC-AUC  :", lr_roc)
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_preds))
print("Classification Report:\n", classification_report(y_test, lr_preds))

# -------------------------------------------------
# 2. Random Forest (Advanced Model)
# -------------------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_preds)
rf_recall = recall_score(y_test, rf_preds)
rf_roc = roc_auc_score(y_test, rf_probs)

print("\n================ Random Forest Results ================")
print("Accuracy :", rf_accuracy)
print("Recall   :", rf_recall)
print("ROC-AUC  :", rf_roc)
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_preds))
print("Classification Report:\n", classification_report(y_test, rf_preds))

# -------------------------------------------------
# Select Best Model (Based on ROC-AUC)
# -------------------------------------------------
if rf_roc > lr_roc:
    best_model = rf_model
    best_model_name = "Random Forest"
else:
    best_model = lr_model
    best_model_name = "Logistic Regression"

# -------------------------------------------------
# Save Best Model & Feature Names
# -------------------------------------------------
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(feature_names, "models/feature_names.pkl")

print("\n===================================================")
print(f"✅ Best Model Selected : {best_model_name}")
print("✅ Model saved to      : models/best_model.pkl")
print("✅ Feature names saved : models/feature_names.pkl")
print("===================================================")
