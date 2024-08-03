"""
This script is used to create a logger using Tensorboard.
"""
from typing import Any, Literal, Optional, List

import numpy
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch

from data_preperation import MNISTDataClasses
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
        self.device_profiler_state = False

    def setup(self, run_name: str | None = None):
        """
        This method sets up the Tensorboard for tracking.
        :param run_name: Name of the run.
        :return: The writer object.
        """
        self.run_name = run_name
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

    def log_model_summary(self, model, hyper_parameters: HyperParameters, device: str):
        """
        This method logs the model summary to tensorboard.
        :param model: Loaded model
        :param hyper_parameters: Hyperparameters for the model
        :param device: Device to run the model on
        """
        _, model_summary = summary(model, input_size=(hyper_parameters.batch_size, hyper_parameters.input_size), device=device)
        self.writer.add_text("Model Summary", text_string=str(model_summary))

    def log_hyper_parameters(self, hyper_parameters: HyperParameters):
        """
        This method logs the hyperparameters to Tensorboard.
        :param hyper_parameters: The hyperparameters.
        """
        self.writer.add_hparams(hyper_parameters.__dict__, {})


    def log_loss(self, loss_type: str, loss: float, step: int):
        """
        This method logs the loss to Tensorboard.
        :param loss_type: The type of loss.
        :param loss: The loss value.
        :param step: The step value
        """
        self.writer.add_scalars("Loss", {loss_type: loss}, step)

    def log_execution_time(self, execution_type: str, time_taken: int, step: int):
        """
        This method logs the execution time to Tensorboard.
        :param execution_type: The type of execution.
        :param time_taken: The time taken.
        :param step: The step value.
        """
        self.writer.add_scalars("Execution Time", {execution_type: time_taken}, step)


    def stop(self):
        """
        This method stops the Tensorboard.
        """
        self.writer.close()
        if self.device_profiler_state:
            self.device_profiler_state = False
            self.profiler.stop()

    def visualize_embeddings(self, features: Any, labels: Any, global_step: int, select_n_points: Optional[int]=100, tag: Optional[str] = "step", classes : MNISTDataClasses | None = None):
        """
        This method visualizes the embeddings.
        :param features: The features.
        :param labels: The labels.
        :param select_n_points: The number of points to select.
        :param global_step: The global step.
        :param tag: The tag.
        :param classes: The class name mapping.
        """
        # convert features & labels to numpy
        features = features.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        # shuffle features & labels randomly; then select N points
        assert len(features) == len(labels), "features and labels must have the same length"
        numpy.random.seed(10)
        permutations = numpy.random.permutation(len(features))
        features = features[permutations]
        labels = labels[permutations]

        # select first N points
        features = features[:select_n_points]
        labels = labels[:select_n_points]

        # what label to show for each data point
        if classes:
            labels = [classes.get_class_name(label) for label in labels]
        else:
            labels = [str(label) for label in labels]

        # IMPORTANT NOTE: The embedding visualization can be seen in the "Projector" tab in Tensorboard.
        # If we choose to visualize by color; all embeddings will be colored as per their class/labels.
        # In the left panel, we can see the class names and their colors.
        # But if we switch on dark mode in visualization; the colors will change. And they will not match the colors in the left panel.
        self.writer.add_embedding(features, metadata=labels, global_step=global_step, tag=tag)

    def add_pr_curves(self, probabilities, groundtruth_labels, global_step, num_classes, tag="test_data_pr_curve"):
        """
        This method adds precision-recall curves to Tensorboard.
        :param probabilities: The probabilities. Make sure that it is probabilities and not predictions.
                              Probabilities are basically softmax of the predictions.
                              They have values in [0,1] and sum to 1.
        :param groundtruth_labels: The ground truth labels.
        :param global_step: The global step.
        :param num_classes: The number of classes.
        :param tag: The tag.
        """
        # convert to numpy
        probabilities = probabilities.cpu().detach().numpy()
        groundtruth_labels = groundtruth_labels.cpu().detach().numpy()

        # plot all the pr curves
        for class_index in range(num_classes):
            # ground truth for class with index class_index. 1 if class_index, 0 otherwise
            groundtruth_for_class = groundtruth_labels == class_index

            # Find the probabilities (for each test data point) for the class with index class_index
            predictions_for_class = probabilities[:, class_index]

            self.writer.add_pr_curve(tag+"_for class "+str(class_index+1),
                                     groundtruth_for_class,
                                predictions_for_class,
                                global_step=global_step)

    def setup_device_profiler(self):
        """
        This method sets up the device profiler.
        """
        if torch.cuda.is_available():
            activities = [torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]
        else:
            activities = [torch.profiler.ProfilerActivity.CPU]
        self.profiler = torch.profiler.profile(
                        activities=activities,
                        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=0),
                        on_trace_ready=torch.profiler.tensorboard_trace_handler(self.log_dir+"/"+self.run_name),
                        record_shapes=False,
                        profile_memory=False,
                        with_stack=False,
                        with_flops=False,
                        with_modules=False,
                        )
        self.profiler.start()

        self.device_profiler_state = True

        return self.profiler

    def device_profiler_step(self):
        """
        This method defines a step boundary for device profiler
        """
        assert self.device_profiler_state, "Device profiler is not set up. Please call setup_device_profiler() method first."
        self.profiler.step()


