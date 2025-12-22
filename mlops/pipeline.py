# pipeline.py
from kfp import dsl
from kfp.components import create_component_from_func


# -------------------------
# Load Data Component
# -------------------------
def load_data_fn(
    X_path: str,
    y_path: str,
):
    from mlops.components.load_data import load_data
    import pandas as pd

    X_data, y_data = load_data()

    pd.DataFrame(X_data).to_csv(X_path, index=False)
    pd.DataFrame(y_data).to_csv(y_path, index=False)


load_data_op = create_component_from_func(
    load_data_fn,
    base_image="asif1993/mlops-training:latest",
)


# -------------------------
# Train Model Component
# -------------------------
def train_fn(
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


train_op = create_component_from_func(
    train_fn,
    base_image="asif1993/mlops-training:latest",
)


# -------------------------
# Evaluate Model Component
# -------------------------
def evaluate_fn(
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

    return evaluate_model(model_obj, X_df, y_df)


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
