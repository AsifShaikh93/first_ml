# train.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train_model(X, y):
    mlflow.set_experiment("diabetes-ct")
    mlflow.set_tracking_uri("https://mlflow-apirule.c-321a6c0.stage.kyma.ondemand.com")

    with mlflow.start_run(run_name="train"):
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = RandomForestClassifier(n_estimators=200,max_depth=10,random_state=42)

        model.fit(X_train, y_train)

        # log model artifact
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=None  # registration happens later
        )

        return model, X_test, y_test
