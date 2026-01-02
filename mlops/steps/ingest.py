import mlflow
from mlops.components.load_data import load_data
import joblib

def ingest():
    X, y = load_data()

    with mlflow.start_run(run_name="ingest"):
        mlflow.log_param("rows", len(X))
        joblib.dump(X, "X.pkl")
        joblib.dump(y, "y.pkl")

if __name__ == "__main__":
    ingest()        

        