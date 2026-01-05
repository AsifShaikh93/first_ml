import mlflow
import joblib
import argparse
from mlops.components.train import train_model
from sklearn.metrics import mean_squared_error

def train(train_run_id: str):
    with mlflow.start_run(run_id=train_run_id):
    
        X = joblib.load("X.pkl")
        y = joblib.load("y.pkl")

        model, X_test, y_test, params = train_model(X, y)
        mlflow.log_params(params)

        y_pred = model.predict(X_test)
        mlflow.log_metrics({"mse": mean_squared_error(y_test, y_pred)})

        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

        # Optional: log test data as artifacts (fine)
        joblib.dump(X_test, "X_test.pkl")
        joblib.dump(y_test, "y_test.pkl")
        # joblib.dump(model, "model.pkl")
        mlflow.log_artifact("X_test.pkl")
        mlflow.log_artifact("y_test.pkl")
        # mlflow.log_artifact("model.pkl", artifact_path="model")
        print("Artifacts:", mlflow.artifacts.list_artifacts(run_id=train_run_id))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_run_id", required=True)
    args = parser.parse_args()

    train(args.train_run_id)
