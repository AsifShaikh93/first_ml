from kfp import dsl
from kfp.dsl import component


@component
def load_data_op():
    from mlops.components.load_data import load_data
    X, y = load_data()
    return X, y


@component
def train_op(X, y):
    from mlops.components.train import train_model
    model, X_test, y_test = train_model(X, y)
    return model, X_test, y_test


@component
def evaluate_op(model, X_test, y_test) -> float:
    from mlops.components.evaluate import evaluate_model
    accuracy = evaluate_model(model, X_test, y_test)
    return accuracy


@component
def save_model_op(model) -> str:
    from mlops.components.save_model import save_model
    model_path = save_model(model)
    return model_path


@component
def compare_and_register_op(
    model_path: str,
    accuracy: float
):
    from mlops.components.compare_and_register import compare_and_register
    compare_and_register(
        model_path=model_path,
        new_accuracy=accuracy
    )


@dsl.pipeline(
    name="diabetes-ct-pipeline",
    description="Continuous training with MLflow model registry"
)
def diabetes_pipeline():
    X, y = load_data_op()
    model, X_test, y_test = train_op(X, y)
    accuracy = evaluate_op(model, X_test, y_test)
    model_path = save_model_op(model)
    compare_and_register_op(model_path, accuracy)
