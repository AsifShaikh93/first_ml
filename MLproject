name: diabetes-mlflow-pipeline

entry_points:
  ingest:
    command: "python mlops/steps/ingest.py"

  train:
    command: "python mlops/steps/train.py"

  evaluate:
    command: "python mlops/steps/evaluate.py"

  register:
    command: "python mlops/steps/register.py"
