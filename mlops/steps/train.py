import mlflow
import joblib
from mlops.components.train import train_model
import os

class SklearnWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model"])

    def predict(self, context, model_input):
        return self.model.predict(model_input)

def train():
     
    with mlflow.start_run(run_name="train") as run:
        
        X=joblib.load ("X.pkl")
        y=joblib.load("y.pkl")
        model, X_test, y_test = train_model(X,y)

        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(model, "artifacts/model.pkl")

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=SklearnWrapper(),
            artifacts={"model": "artifacts/model.pkl"}
        )
        
        print(f"TRAIN_RUN_ID={run.info.run_id}")

if __name__ == "__main__":
    train()        