import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os


def save_model(model) -> str:
    MODEL_NAME = "diabetes-model"
    EXPERIMENT_NAME = "diabetes-ct"

    mlflow.set_tracking_uri('https://mlflow-apirule.c-321a6c0.stage.kyma.ondemand.com')
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run: 
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        model_uri = f"runs:/{run.info.run_id}/model"
        print(f"Model registered at: {model_uri}")

    return model_uri
