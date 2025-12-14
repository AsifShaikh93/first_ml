# load_data.py
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/refs/heads/master/diabetes.csv"

FEATURE_COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]
TARGET_COLUMN = "Outcome"


def load_data():
    df = pd.read_csv(DATA_URL)
    print("Columns:", df.columns.tolist())

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    return X, y

