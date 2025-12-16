import mlflow
from sklearn.metrics import accuracy_score


def evaluate_model(model, X_test, y_test):
    accuracy = accuracy_score(y_test, model.predict(X_test))

    mlflow.log_metric("accuracy", accuracy)

    print(f"Model accuracy: {accuracy:.4f}")
    return accuracy
