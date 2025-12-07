import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


DEFAULT_URL = "https://raw.githubusercontent.com/plotly/datasets/refs/heads/master/diabetes.csv"


def train_and_save_model(csv_url: str = DEFAULT_URL, output_path: str = "diabetes_model.pkl"):
    """Train a RandomForest model and save it to output_path."""
    print(f"[train_logic] Loading data from {csv_url}")
    df = pd.read_csv(csv_url)
    print("[train_logic] Columns:", df.columns.tolist())

    X = df[["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]]
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"[train_logic] Test accuracy: {accuracy:.4f}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    joblib.dump(model, output_path)
    print(f"[train_logic] Model saved at {output_path}")

    return accuracy, output_path
