import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os


def save_model(model) -> str:
    """
    Logs and registers a model in MLflow Model Registry.

    Returns:
        model_uri (str): MLflow model URI (used for comparison/registration)
    """

    # ---- CONFIG (adjust only if needed) ----
    MODEL_NAME = "diabetes-model"
    EXPERIMENT_NAME = "diabetes-ct"

    # Optional: set explicitly if not injected via env
    # mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

    # ---------------------------------------

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        # Log the model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        model_uri = f"runs:/{run.info.run_id}/model"

    return model_uri
