import h2o
import os
import mlflow
from h2o.automl import H2OAutoML

def run_automl():
    h2o.init()

    # Load latest uploaded dataset
    folder = "uploaded_datasets"
    files = os.listdir(folder)
    if not files:
        raise FileNotFoundError("No dataset found")

    latest_file = max([os.path.join(folder, f) for f in files], key=os.path.getctime)
    df = h2o.import_file(latest_file)

    # Assume last column is target
    target = df.col_names[-1]
    x = df.col_names[:-1]

    # Train AutoML
    aml = H2OAutoML(max_models=5, seed=1)
    aml.train(x=x, y=target, training_frame=df)

    # Log model
    with mlflow.start_run(run_name="AutoML Training") as run:
        mlflow.set_tag("model_type", "H2OAutoML")
        mlflow.h2o.log_model(aml.leader, "model")

    # Return leaderboard
    lb = aml.leaderboard.as_data_frame()
    return lb.to_dict(orient="records")
