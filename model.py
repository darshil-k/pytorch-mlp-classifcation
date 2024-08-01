"""
This file is used to define the model architecture.
"""
from typing import Optional

# Importing the required libraries
import torch
import torch.nn as nn
from pydantic import BaseModel

from hyper_parameters import HyperParameters


class ModelParameters(BaseModel):
    hidden_size: Optional[int] = 500

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    """
    This class is used to define the neural network model.
    """
    def __init__(self, hyper_parameters: HyperParameters):
        """
        This method initializes the class.
        :param hyper_parameters: The hyperparameters for the model.
        """
        super(NeuralNet, self).__init__()
        model_parameters = ModelParameters()

        self.fc1 = nn.Linear(hyper_parameters.input_size, model_parameters.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(model_parameters.hidden_size, hyper_parameters.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This method defines the forward pass of the neural network.
        :param x: The input tensor.
        :return: The output tensor.
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out