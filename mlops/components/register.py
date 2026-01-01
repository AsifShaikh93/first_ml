# mlops/components/evaluate.py

import mlflow

MODEL_NAME = "diabetes-model"


def compare_and_register(
    new_accuracy: float,
    threshold: float = 0.0
):
    client = mlflow.tracking.MlflowClient()
    mlflow.set_tracking_uri("https://mlflow-apirule.c-321a6c0.stage.kyma.ondemand.com")

    
    try:
        versions = client.search_model_versions(
            f"name='{MODEL_NAME}'"
        )
    except Exception:
        versions = []

    if versions:
        latest = max(
            versions,
            key=lambda v: int(v.version)
        )
        
        prev_accuracy = float(latest.tags["accuracy"])

        if new_accuracy <= prev_accuracy:
            print(
                f"Model rejected: {new_accuracy:.4f} "
                f"<= {prev_accuracy:.4f}"
            )
            return False

    print("Registering new model in MLflow...")

    run = mlflow.active_run()
    model_uri = f"runs:/{run.info.run_id}/model"

    result = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    client.set_model_version_tag(
        name=MODEL_NAME,
        version=result.version,
        key="accuracy",
        value=str(new_accuracy)
    )

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=result.version,
        stage="Production"
    )

    print("Model promoted to Production")
    return True
