from kfp import dsl
from kfp.dsl import component, Dataset, Model, Output


# -------------------------
# Load Data Component
# -------------------------
@component(base_image="python:3.10-slim")
def load_data_op(
    X: Output[Dataset],
    y: Output[Dataset],
):
    from mlops.components.load_data import load_data
    import pandas as pd

    X_data, y_data = load_data()

    pd.DataFrame(X_data).to_csv(X.path, index=False)
    pd.DataFrame(y_data).to_csv(y.path, index=False)


# -------------------------
# Train Model Component
# -------------------------
@component(base_image="python:3.10-slim")
def train_op(
    X: Dataset,
    y: Dataset,
    model: Output[Model],
    X_test: Output[Dataset],
    y_test: Output[Dataset],
):
    from mlops.components.train import train_model
    import pandas as pd
    import joblib

    X_df = pd.read_csv(X.path)
    y_df = pd.read_csv(y.path)

    trained_model, X_test_data, y_test_data = train_model(X_df, y_df)

    joblib.dump(trained_model, model.path)
    pd.DataFrame(X_test_data).to_csv(X_test.path, index=False)
    pd.DataFrame(y_test_data).to_csv(y_test.path, index=False)


# -------------------------
# Evaluate Model Component
# -------------------------
@component(base_image="python:3.10-slim")
def evaluate_op(
    model: Model,
    X_test: Dataset,
    y_test: Dataset,
) -> float:
    from mlops.components.evaluate import evaluate_model
    import pandas as pd
    import joblib

    model_obj = joblib.load(model.path)
    X_df = pd.read_csv(X_test.path)
    y_df = pd.read_csv(y_test.path)

    accuracy = evaluate_model(model_obj, X_df, y_df)
    return accuracy


# -------------------------
# Save Model Component
# -------------------------
@component(base_image="python:3.10-slim")
def save_model_op(model: Model) -> str:
    from mlops.components.save_model import save_model
    import joblib

    model_obj = joblib.load(model.path)
    model_path = save_model(model_obj)
    return model_path


# -------------------------
# Compare & Register Component
# -------------------------
@component(base_image="python:3.10-slim")
def compare_and_register_op(
    model_path: str,
    accuracy: float,
):
    from mlops.components.compare_and_register import compare_and_register

    compare_and_register(
        model_path=model_path,
        new_accuracy=accuracy,
    )


# -------------------------
# Pipeline Definition
# -------------------------
@dsl.pipeline(
    name="diabetes-ct-pipeline",
    description="Continuous training with MLflow model registry",
)
def diabetes_pipeline():
    load_task = load_data_op()

    train_task = train_op(
        X=load_task.outputs["X"],
        y=load_task.outputs["y"],
    )

    accuracy = evaluate_op(
        model=train_task.outputs["model"],
        X_test=train_task.outputs["X_test"],
        y_test=train_task.outputs["y_test"],
    )

    model_path = save_model_op(
        model=train_task.outputs["model"]
    )

    compare_and_register_op(
        model_path=model_path.output,
        accuracy=accuracy.output,
    )
