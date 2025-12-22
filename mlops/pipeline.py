# pipeline.py
from kfp import dsl
from kfp.components import create_component_from_func


# -------------------------
# Load Data Component
# -------------------------
def load_data_fn():
    from mlops.components.load_data import load_data
    import pandas as pd

    X_data, y_data = load_data()

    pd.DataFrame(X_data).to_csv("/tmp/X.csv", index=False)
    pd.DataFrame(y_data).to_csv("/tmp/y.csv", index=False)


load_data_op = create_component_from_func(
    load_data_fn,
    base_image="asif1993/mlops-training:latest",
    file_outputs={
        "X": "/tmp/X.csv",
        "y": "/tmp/y.csv",
    },
)


# -------------------------
# Train Model Component
# -------------------------
def train_fn(X_path: str, y_path: str):
    from mlops.components.train import train_model
    import pandas as pd
    import joblib

    X_df = pd.read_csv(X_path)
    y_df = pd.read_csv(y_path)

    model, X_test, y_test = train_model(X_df, y_df)

    joblib.dump(model, "/tmp/model.joblib")
    pd.DataFrame(X_test).to_csv("/tmp/X_test.csv", index=False)
    pd.DataFrame(y_test).to_csv("/tmp/y_test.csv", index=False)


train_op = create_component_from_func(
    train_fn,
    base_image="asif1993/mlops-training:latest",
    file_outputs={
        "model": "/tmp/model.joblib",
        "X_test": "/tmp/X_test.csv",
        "y_test": "/tmp/y_test.csv",
    },
)


# -------------------------
# Evaluate Model Component
# -------------------------
def evaluate_fn(model_path: str, X_test_path: str, y_test_path: str) -> float:
    from mlops.components.evaluate import evaluate_model
    import pandas as pd
    import joblib

    model = joblib.load(model_path)
    X_df = pd.read_csv(X_test_path)
    y_df = pd.read_csv(y_test_path)

    return evaluate_model(model, X_df, y_df)


evaluate_op = create_component_from_func(
    evaluate_fn,
    base_image="asif1993/mlops-training:latest",
)


# -------------------------
# Compare & Register Component
# -------------------------
def compare_and_register_fn(accuracy: float):
    from mlops.components.compare_and_register import compare_and_register
    compare_and_register(new_accuracy=accuracy)


compare_and_register_op = create_component_from_func(
    compare_and_register_fn,
    base_image="asif1993/mlops-training:latest",
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
        X_path=load_task.outputs["X"],
        y_path=load_task.outputs["y"],
    )

    eval_task = evaluate_op(
        model_path=train_task.outputs["model"],
        X_test_path=train_task.outputs["X_test"],
        y_test_path=train_task.outputs["y_test"],
    )

    compare_and_register_op(
        accuracy=eval_task.output
    )
