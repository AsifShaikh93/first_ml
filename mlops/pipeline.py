# pipeline.py
from kfp import dsl
from kfp.dsl import component


# -------------------------
# Load Data Component
# -------------------------
@component(base_image="asif1993/mlops-training:latest")
def load_data_op(
    X_path: str,
    y_path: str,
):
    from mlops.components.load_data import load_data
    import pandas as pd

    X_data, y_data = load_data()

    pd.DataFrame(X_data).to_csv(X_path, index=False)
    pd.DataFrame(y_data).to_csv(y_path, index=False)


# -------------------------
# Train Model Component
# -------------------------
@component(base_image="asif1993/mlops-training:latest")
def train_op(
    X_path: str,
    y_path: str,
    model_path: str,
    X_test_path: str,
    y_test_path: str,
):
    from mlops.components.train import train_model
    import pandas as pd
    import joblib

    X_df = pd.read_csv(X_path)
    y_df = pd.read_csv(y_path)

    trained_model, X_test_data, y_test_data = train_model(X_df, y_df)

    joblib.dump(trained_model, model_path)
    pd.DataFrame(X_test_data).to_csv(X_test_path, index=False)
    pd.DataFrame(y_test_data).to_csv(y_test_path, index=False)


# -------------------------
# Evaluate Model Component
# -------------------------
@component(base_image="asif1993/mlops-training:latest")
def evaluate_op(
    model_path: str,
    X_test_path: str,
    y_test_path: str,
) -> float:
    from mlops.components.evaluate import evaluate_model
    import pandas as pd
    import joblib

    model_obj = joblib.load(model_path)
    X_df = pd.read_csv(X_test_path)
    y_df = pd.read_csv(y_test_path)

    accuracy = evaluate_model(model_obj, X_df, y_df)
    return accuracy


# -------------------------
# Compare & Register Component
# -------------------------
@component(base_image="asif1993/mlops-training:latest")
def compare_and_register_op(accuracy: float):
    from mlops.components.compare_and_register import compare_and_register
    compare_and_register(new_accuracy=accuracy)


# -------------------------
# Pipeline Definition (KFP v1)
# -------------------------
@dsl.pipeline(
    name="diabetes-ct-pipeline",
    description="Continuous training with MLflow model registry",
)
def diabetes_pipeline():

    load_task = load_data_op(
        X_path="/tmp/X.csv",
        y_path="/tmp/y.csv",
    )

    train_task = train_op(
        X_path=load_task.outputs["X_path"],
        y_path=load_task.outputs["y_path"],
        model_path="/tmp/model.joblib",
        X_test_path="/tmp/X_test.csv",
        y_test_path="/tmp/y_test.csv",
    )

    eval_task = evaluate_op(
        model_path=train_task.outputs["model_path"],
        X_test_path=train_task.outputs["X_test_path"],
        y_test_path=train_task.outputs["y_test_path"],
    )

    compare_and_register_op(
        accuracy=eval_task.output
    )
