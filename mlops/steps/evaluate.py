# mlops/steps/evaluate.py

import mlflow
import joblib
from sklearn.metrics import accuracy_score
import sys

train_run_id = sys.argv[2]

def evaluate(train_run_id: str):
    with mlflow.start_run(run_name="evaluate", nested=True):

        model_uri = f"runs:/{train_run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)

        X_test = joblib.load("X_test.pkl")
        y_test = joblib.load("y_test.pkl")

        accuracy = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("accuracy", accuracy)

        print(f"EVAL_ACCURACY={accuracy}")

if __name__ == "__main__":
    evaluate(train_run_id)        
        