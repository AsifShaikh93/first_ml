from kfp import dsl
from kfp.dsl import component


@component
def load_data_op() -> str:
    from mlops.components.load_data import load_data
    import joblib
    import os

    X, y = load_data()
    os.makedirs("/tmp/data", exist_ok=True)

    data_path = "/tmp/data/data.pkl"
    joblib.dump((X, y), data_path)

    return data_path


@component
def train_op(data_path: str) -> str:
    from mlops.components.train import train_model
    import joblib
    import os

    X, y = joblib.load(data_path)
    model, X_test, y_test = train_model(X, y)

    os.makedirs("/tmp/model", exist_ok=True)
    model_path = "/tmp/model/model.pkl"
    test_path = "/tmp/model/test.pkl"

    joblib.dump(model, model_path)
    joblib.dump((X_test, y_test), test_path)

    return model_path


@component
def evaluate_op(model_path: str, data_path: str) -> float:
    from mlops.components.evaluate import evaluate_model
    import joblib

    model = joblib.load(model_path)
    X, y = joblib.load(data_path)

    # re-split to get test set (same random_state as training)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    accuracy = evaluate_model(model, X_test, y_test)
    return accuracy


@component
def register_model_op(model_path: str, accuracy: float):
    from mlops.components.compare_and_register import compare_and_register
    compare_and_register(
        model_path=model_path,
        new_accuracy=accuracy
    )


@dsl.pipeline(
    name="diabetes-ct-pipeline",
    description="Continuous training with MLflow Model Registry"
)
def diabetes_pipeline():
    data_path = load_data_op()
    model_path = train_op(data_path)
    accuracy = evaluate_op(model_path, data_path)
    register_model_op(model_path, accuracy)
