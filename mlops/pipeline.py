# pipeline.py
from kfp import dsl
from kfp.dsl import component, Input, Output


# -------------------------
# Load Data Component
# -------------------------
@component(base_image="asif1993/mlops-training:latest")
def load_data_op(
    X: Output[dsl.Dataset],
    y: Output[dsl.Dataset],
):
    from mlops.components.load_data import load_data
    import pandas as pd

    X_data, y_data = load_data()

    pd.DataFrame(X_data).to_csv(X.path, index=False)
    pd.DataFrame(y_data).to_csv(y.path, index=False)


# -------------------------
# Train Model Component
# -------------------------
@component(base_image="asif1993/mlops-training:latest")
def train_op(
    X: Input[dsl.Dataset],
    y: Input[dsl.Dataset],
    model: Output[dsl.Model],
    X_test: Output[dsl.Dataset],
    y_test: Output[dsl.Dataset],
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
@component(base_image="asif1993/mlops-training:latest")
def evaluate_op(
    model: Input[dsl.Model],
    X_test: Input[dsl.Dataset],
    y_test: Input[dsl.Dataset],
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

    load_task = load_data_op()

    train_task = train_op(
        X=load_task.outputs["X"],
        y=load_task.outputs["y"],
    )

    eval_task = evaluate_op(
        model=train_task.outputs["model"],
        X_test=train_task.outputs["X_test"],
        y_test=train_task.outputs["y_test"],
    )

    compare_and_register_op(
        accuracy=eval_task.output
    )
