import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "diabetes-model"

def register():
    client = MlflowClient()
    run = mlflow.active_run()

    model_uri = f"runs:/{run.info.run_id}/model"

    result = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=result.version,
        stage="Production"
    )

    print(f"Model {MODEL_NAME} v{result.version} promoted to Production")