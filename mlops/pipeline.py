name: diabetes-mlflow-pipeline

entry_points:
  ingest:
    command: "python mlops/steps/ingest.py"

  train:
    command: "python mlops/steps/train.py"

  evaluate:
    parameters: 
      train_run_id: {type: string}
    command: "python mlops/steps/evaluate.py {train_run_id}"

  register:
    parameters: 
      train_run_id: {type: string}
    command: "python mlops/steps/register.py {train_run_id}"
