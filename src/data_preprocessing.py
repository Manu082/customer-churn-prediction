# src/data_preprocessing.py

import pandas as pd

def load_data(path):
    """
    Load customer churn dataset
    """
    df = pd.read_csv(path)
    return df


def basic_data_info(df):
    """
    Display basic information about the dataset
    """
    print("\n--- First 5 Rows ---")
    print(df.head())

    print("\n--- Dataset Shape ---")
    print(df.shape)

    print("\n--- Column Names ---")
    print(df.columns)

    print("\n--- Data Types ---")
    print(df.dtypes)

    print("\n--- Missing Values ---")
    print(df.isnull().sum())


if __name__ == "__main__":
    data_path = "data/raw/telecom_churn.csv"
    df = load_data(data_path)
    basic_data_info(df)
