import mlflow
import joblib
from mlops.components.train import train_model
from mlops.steps.ingest import ingest

def train():
    with mlflow.start_run(run_name="train") as run:
        X,y=ingest()
        model, X_test, y_test = train_model(X,y)

        mlflow.sklearn.log_model(model, "model")
        joblib.dump(X_test, "X_test.pkl")
        joblib.dump(y_test, "y_test.pkl")

        mlflow.log_artifact("X_test.pkl")
        mlflow.log_artifact("y_test.pkl")
        
        print(f"TRAIN_RUN_ID={run.info.run_id}")

if __name__ == "__main__":
    train()        