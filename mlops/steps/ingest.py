import mlflow
from mlops.components.load_data import load_data

def ingest():
    X, y = load_data()

    with mlflow.start_run(run_name="ingest"):
        mlflow.log_param("rows", len(X))

        