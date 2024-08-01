"""
This script is used to create a logger using MLFlow.
"""
from typing import Any

# Importing the required libraries
import mlflow
from pydantic import BaseModel
from torchsummary.torchsummary import summary
import os

from hyper_parameters import HyperParameters


class MLFlow_params(BaseModel):
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "mlp-classification-mnist"

class MLFlowLogging:
    """
    This class is used to create a logger using MLFlow.
    """
    def __init__(self, run_name: str | None = None):
        """
        This method initializes the class.
        :param run_name: The name of the run.
        """
        self.tracking_uri = MLFlow_params().tracking_uri
        self.experiment_name = MLFlow_params().experiment_name
        self.run_name = run_name
        self.setup()

    def setup(self):
        """
        This method sets up the MLFlow for tracking.
        """

        try:
            mlflow.set_tracking_uri(self.tracking_uri)
        except Exception as e:
            raise Exception(f"Connection to MLFlow server failed. Please check the server is started on URI: {self.tracking_uri}"
                            f"More help to start the server: [CLI] mlflow ui --help")
        mlflow.set_experiment(self.experiment_name)

        mlflow.start_run(run_name=self.run_name)

    def log_params(self, params_and_values: dict):
        """
        This method logs the parameters to MLFlow.
        :param params_and_values: A dictionary containing the parameters and their values.
        """
        mlflow.log_params(params_and_values)

    def log_artifact(self, artifact_path: str):
        """
        This method logs the artifact to MLFlow.
        :param artifact_path: The path of the artifact.
        """
        mlflow.log_artifact(artifact_path)

    def log_model_summary(self, model, hyper_parameters: HyperParameters):
        """
        This method logs the model summary to MLFlow.
        :param model: Loaded model
        :param hyper_parameters: Hyperparameters for the model
        """
        f = open("model_summary.txt", "w", encoding="utf-8")
        model_summary = summary(model, input_size=(hyper_parameters.batch_size, hyper_parameters.input_size))
        f.write(str(model_summary))
        f.close()
        mlflow.log_artifact("model_summary.txt")
        os.remove("model_summary.txt")

    def log_metric(self, graph_name: str, metric_value: Any, x_axis_value: int):
        """
        This method logs the metric to MLFlow.
        :param graph_name: Name of the graph.
        :param metric_value: Numeric value of the metric.
        :param x_axis_value: step value on x-axis.
        :return:
        """
        mlflow.log_metric(graph_name, metric_value, step=x_axis_value)


    def stop(self):
        """
        This method stops the run.
        """
        mlflow.end_run()