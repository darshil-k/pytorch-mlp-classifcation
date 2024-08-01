"""
This file is used to test the ml_flow.py file.
"""

# Importing the required libraries
import mlflow

from ml_flow_utils import MLFlow_params


def test_ml_flow():
    """
    This method is used to test the MLFlow class.
    :return:
    """
    try:
        mlflow.set_tracking_uri(MLFlow_params.TRACKING_URI.value)
    except Exception as e:
        raise Exception(f"Connection to MLFlow server failed. Please check the server is started on URI: {MLFlow_params.TRACKING_URI.value}"
                        f"More help to start the server: [CLI] mlflow ui --help")
    mlflow.set_experiment(MLFlow_params.experiment_name.value)
    mlflow.start_run(run_name="Test Run")
    mlflow.log_param("param1", 10)
    mlflow.log_metric("metric1", 20)
    mlflow.end_run()


