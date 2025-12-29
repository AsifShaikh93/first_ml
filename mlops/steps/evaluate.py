import mlflow
import joblib
from sklearn.metrics import accuracy_score

def evaluate():
    with mlflow.start_run(run_name="evaluate"):
        model = mlflow.sklearn.load_model("runs:/../train/model")

        X_test = joblib.load("X_test.pkl")
        y_test = joblib.load("y_test.pkl")

        accuracy = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("accuracy", accuracy)

        print(f"Accuracy: {accuracy:.4f}")