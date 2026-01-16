# 📊 **Customer Churn Prediction – Data Analytics & Machine Learning**

---

## 🔍 **1. Project Overview**

Customer churn is a major challenge for **service-based industries** such as **telecom, banking, and SaaS companies**.

This project focuses on **analyzing customer behavior data** to:

- **Identify churn patterns**
- **Understand key risk factors**
- **Predict the likelihood of customer churn**

The project combines:

- **Data Analytics**
- **Exploratory Data Analysis (EDA)**
- **Machine Learning**

and is deployed as an **interactive Streamlit dashboard** for **real-time analysis and prediction**.

---

## 🎯 **2. Objectives**

1. **Analyze customer data**
   - Identify churn trends
   - Understand customer behavior patterns

2. **Perform segment-wise data analytics**
   - Contract type
   - Payment method
   - Customer tenure
   - Monthly and total charges

3. **Predict customer churn probability**
   - Use machine learning classification models

4. **Present insights visually**
   - Interactive dashboard
   - Business-friendly charts

5. **Support business decision-making**
   - Customer retention strategies
   - Risk-based targeting

---

## 🧠 **3. Key Features**

✔ **Data cleaning and preprocessing**  
✔ **Exploratory Data Analysis (EDA)**  
✔ **Segment-wise churn analysis**  
✔ **Interactive filters**
- Contract Type
- Payment Method  

✔ **Churn probability prediction**  
✔ **Risk categorization**
- **Low Risk**
- **Medium Risk**
- **High Risk**

✔ **Deployed web application using Streamlit**

---

## 📂 **4. Project Structure**

customer-churn-prediction/
│
├── app.py # Streamlit application
├── requirements.txt # Project dependencies
├── README.md # Project documentation
├── .gitignore
│
├── data/
│ ├── raw/
│ │ └── telecom_churn.csv
│ └── processed/
│ └── cleaned_churn_data.csv
│
├── models/
│ ├── best_model.pkl # Trained ML model
│ ├── scaler.pkl # Feature scaler
│ └── feature_names.pkl # Model feature list
│
├── src/
│ ├── data_preprocessing.py # Data cleaning
│ ├── feature_engineering.py # Encoding & scaling
│ └── model_training.py # Model training & evaluation
│
├── notebooks/ # Jupyter notebooks (EDA & experiments)
└── reports/ # Analysis reports & insights

---

## 📊 **5. Data Analytics Approach**

### **5.1 Data Cleaning & Preparation**
- Converted **inconsistent data types**
- Handled **missing values**
- Removed **non-analytical identifiers**
- Encoded **categorical variables**

---

### **5.2 Exploratory Data Analysis (EDA)**
- **Churn distribution analysis**
- **Contract type vs churn**
- **Monthly charges vs churn**
- **Tenure-based churn trends**
- **Segment-wise analytics using interactive filters**

---

### **5.3 Business Insights**
- **Month-to-month customers** show **higher churn**
- Customers with **higher monthly charges** are **more likely to churn**
- **Long-tenure customers** have **higher retention rates**

---

## 🤖 **6. Machine Learning Models Used**

- **Logistic Regression**
  - Used as a **baseline model**

- **Random Forest Classifier**
  - Selected as the **final model** due to better performance

---

### 📈 **6.1 Model Evaluation Metrics**
- **Accuracy**
- **Recall**
- **ROC-AUC Score**

👉 The **Random Forest model** achieved superior performance and was selected for deployment.

---

## 🖥️ **7. Interactive Dashboard (Streamlit)**

The Streamlit dashboard allows users to:

- **Input customer details**
- **Predict churn probability**
- **View churn risk levels**
- **Analyze churn patterns dynamically**
- **Visualize insights through charts**
- **Filter data segment-wise for analytics**

---

## 🛠️ **8. Tech Stack**

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Web Application:** Streamlit
- **Version Control:** Git & GitHub

---

## 🙌 **09. Acknowledgements**

- **Dataset inspired by telecom churn data**
- **Open-source Python and Streamlit community**

## 📬 **10. Contact**

- **Name:** **Manu Didwania**
- **GitHub:** **https://github.com/Manu082**

---

## 🚀 **11. How to Run Locally**

```bash
git clone https://github.com/Manu082/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
streamlit run app.py
