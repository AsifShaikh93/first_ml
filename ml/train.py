import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

url="https://raw.githubusercontent.com/plotly/datasets/refs/heads/master/diabetes.csv"

df=pd.read_csv(url)

print("columns:", df.columns.tolist())

X=df[["Pregnancies","Glucose", "BloodPressure", "BMI", "Age"]]
y=df["Outcome"]

X_train, X_test, y_train, y_test= train_test_split( X, y, test_size=0.2, random_state=42)

model= RandomForestClassifier()
model.fit(X_train, y_train)

MODEL_DIR = "/mnt/models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "diabetes_model.pkl")

# === 5) Save model to PVC ===
joblib.dump(model, MODEL_PATH)
print(f"Model saved at {MODEL_PATH}")

# === 6) Optional: Verify model exists ===
if os.path.exists(MODEL_PATH):
    print("Verified: model file exists in PVC.")
else:
    print("ERROR: model file not found after saving!")