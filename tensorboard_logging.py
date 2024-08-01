"""
This script is used to create a logger using Tensorboard.
"""
from typing import Any, Literal

import torchvision
from torch.utils.tensorboard import SummaryWriter
from model_utils import summary

from hyper_parameters import HyperParameters


class TensorboardLogging:
    """
    This class is used to create a logger using Tensorboard.
    """
    def __init__(self, log_dir: str = "tensorboard_logs", run_name: str | None = None):
        """
        This method initializes the class.
        :param log_dir: The directory where the logs will be saved.
        :param run_name: The name of the run.
        """
        self.log_dir = log_dir
        self.setup(run_name)

    def setup(self, run_name: str | None = None):
        """
        This method sets up the Tensorboard for tracking.
        :param run_name: Name of the run.
        :return: The writer object.
        """
        self.writer = SummaryWriter(log_dir=self.log_dir+"/"+run_name)
        return self.writer

    def log_sample_images(self, images: list[Any]):
        """
        This method logs the sample images to Tensorboard.
        :param images: The list of images.
        """
        # create grid of images
        image_grid = torchvision.utils.make_grid(images)
        self.writer.add_image("Sample Images", image_grid)

    def log_model_graph(self, model: Any):
        """
        This method logs the model graph to Tensorboard.
        :param model: The model.
        """
        self.writer.add_graph(model)

    def log_model_summary(self, model, hyper_parameters: HyperParameters):
        """
        This method logs the model summary to tensorboard.
        :param model: Loaded model
        :param hyper_parameters: Hyperparameters for the model
        """
        _, model_summary = summary(model, input_size=(hyper_parameters.batch_size, hyper_parameters.input_size))
        self.writer.add_text("Model Summary", text_string=str(model_summary))

    def log_hyper_parameters(self, hyper_parameters: HyperParameters):
        """
        This method logs the hyperparameters to Tensorboard.
        :param hyper_parameters: The hyperparameters.
        """
        self.writer.add_hparams(hyper_parameters.__dict__, {})


    def log_loss(self, loss_type: Literal["training","validation"], loss: float, step: int):
        """
        This method logs the loss to Tensorboard.
        :param loss_type: The type of loss. It can be training or validation.
        :param loss: The loss value.
        :param step: The step value
        """
        self.writer.add_scalars("Loss", {loss_type: loss}, step)

    def log_artifact(self, artifact_path: str):
        """
        This method logs the artifact to Tensorboard.
        :param artifact_path: The path of the artifact.
        """
        pass

    def stop(self):
        """
        This method stops the Tensorboard.
        """
        self.writer.close()