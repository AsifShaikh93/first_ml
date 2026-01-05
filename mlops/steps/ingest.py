import mlflow
from mlops.components.load_data import load_data
import joblib
import argparse

def ingest():
    X, y, DATA_URL, FEATURE_COLUMNS, TARGET_COLUMN = load_data()

    mlflow.start_run(run_name="ingest") 
    mlflow.log_param("rows", len(X))
    mlflow.log_param("DATA_URL", DATA_URL)
    mlflow.log_param("FEATURE_COLUMNS", FEATURE_COLUMNS)
    mlflow.log_param("TARGET_COLUMN", TARGET_COLUMN)
    joblib.dump(X, "X.pkl")
    joblib.dump(y, "y.pkl")

    print(f"TRAIN_RUN_ID={mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    ingest()        

        