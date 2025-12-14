
# get_previous_model.py

from kfp.registry import ModelRegistryClient

MODEL_NAME = "diabetes-randomforest"


def get_previous_model_accuracy():
    client = ModelRegistryClient()

    models = client.list_models(filter=f"name={MODEL_NAME}")

    if not models:
        print("No existing model found in registry.")
        return None

    # Get latest version
    latest_model = sorted(
        models,
        key=lambda m: m.create_time,
        reverse=True
    )[0]

    accuracy = latest_model.metrics.get("accuracy")

    print(f"Previous model accuracy: {accuracy}")
    return accuracy
