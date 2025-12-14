
# evaluate.py
from sklearn.metrics import accuracy_score


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Model accuracy: {accuracy:.4f}")

    return accuracy
