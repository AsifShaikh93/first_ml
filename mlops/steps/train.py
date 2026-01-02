import mlflow
import joblib
from mlops.components.train import train_model
import os

def train():
     
    with mlflow.start_run(run_name="train") as run:
        
        X=joblib.load ("X.pkl")
        y=joblib.load("y.pkl")
        model, X_test, y_test = train_model(X,y)

        os.makedirs("model", exist_ok=True)
        joblib.dump(model, "model/model.pkl")

        mlflow.log_artifacts("model", artifact_path="model")
        
        print(f"TRAIN_RUN_ID={run.info.run_id}")

if __name__ == "__main__":
    train()        