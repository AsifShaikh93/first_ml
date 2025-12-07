from kfp import dsl, compiler

@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "scikit-learn", "joblib"],
)
def train_diabetes_model(csv_url: str, model_dir: str):
    import pandas as pd, joblib, os
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    df = pd.read_csv(csv_url)
    X = df[["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]]
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "diabetes_model.pkl")
    joblib.dump(model, path)
    print("Saved model to:", path)

@dsl.pipeline(
    name="diabetes-training-pvc-pipeline",
    description="Train diabetes model and save to PVC"
)
def diabetes_pipeline(
    csv_url: str = "https://raw.githubusercontent.com/plotly/datasets/refs/heads/master/diabetes.csv"
):
    # VolumeOp pointing to existing PVC
    vop = dsl.VolumeOp(
        name="use-existing-pvc",
        resource_name="diabetes-model-pvc",
        modes=["ReadWriteOnce"],
        size="1Gi",
    )

    train_task = train_diabetes_model(
        csv_url=csv_url,
        model_dir="/mnt/models",
    )

    train_task.add_pvolumes({
        "/mnt/models": vop.volume
    })

if __name__ == "__main__":
    compiler.Compiler().compile(
        diabetes_pipeline,
        "diabetes_training_pipeline.yaml"
    )
