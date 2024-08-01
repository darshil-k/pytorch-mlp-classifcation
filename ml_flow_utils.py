"""
This file contains the parameters for the ML Flow.
"""
from enum import Enum
import mlflow

class MLFlow_params(Enum):

    TRACKING_URI = "http://localhost:5000"
    experiment_name = "mlp-classification-mnist"

def set_up_mlflow():
    """
    This method sets up the MLFlow for tracking.
    :return: MLFlow object.
    """

    try:
        mlflow.set_tracking_uri(MLFlow_params.TRACKING_URI.value)
    except Exception as e:
        raise Exception(f"Connection to MLFlow server failed. Please check the server is started on URI: {MLFlow_params.TRACKING_URI.value}"
                        f"More help to start the server: [CLI] mlflow ui --help")
    mlflow.set_experiment(MLFlow_params.experiment_name.value)

    return mlflow
