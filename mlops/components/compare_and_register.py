# compare_and_register.py

from kfp.registry import ModelRegistryClient

MODEL_NAME = "diabetes-randomforest"


def compare_and_register(
    model,
    new_accuracy,
    previous_accuracy,
    model_path
):
    client = ModelRegistryClient()

    if previous_accuracy is not None and new_accuracy <= previous_accuracy:
        print(
            f"New model rejected: "
            f"{new_accuracy:.4f} <= {previous_accuracy:.4f}"
        )
        return False

    print("New model is better. Registering...")

    client.register_model(
        name=MODEL_NAME,
        uri=model_path,
        metrics={"accuracy": new_accuracy},
        description="RandomForest diabetes classifier",
        version_notes="Auto-promoted by Kubeflow pipeline"
    )

    print("Model registered successfully.")
    return True
