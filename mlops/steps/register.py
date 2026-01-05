# mlops/steps/register.py
import argparse
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "diabetes-model"

def register(train_run_id: str):
    with mlflow.start_run(run_id=train_run_id):

        client = MlflowClient()
        model_uri = f"runs:/{train_run_id}/model"
        result = mlflow.register_model(model_uri, MODEL_NAME)

        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="Production",
            version=result.version
        )

        print(f"model_uri={model_uri}")
        print(f"MODEL_NAME={MODEL_NAME}")
        
        print(f"MODEL_URI=models:/{MODEL_NAME}/Production")

        model_version = client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias="Production"
        )

        print("Production model version:", model_version.version)
        print("Run ID:", model_version.run_id)
        print("Source:", model_version.source)

        mlflow.end_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_run_id", required=True)
    args = parser.parse_args()

    register(args.train_run_id)
