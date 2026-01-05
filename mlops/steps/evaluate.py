# mlops/steps/evaluate.py

import mlflow
import joblib
import argparse
from sklearn.metrics import accuracy_score
import sys

def evaluate(train_run_id: str):
    with mlflow.start_run(run_name="ingest"):

        model_uri = f"runs:/{train_run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)

        X_test = joblib.load("X_test.pkl")
        y_test = joblib.load("y_test.pkl")

        accuracy = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("accuracy", accuracy)

        print(f"EVAL_ACCURACY={accuracy}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_run_id", required=True)
    args = parser.parse_args()

    evaluate(args.train_run_id)