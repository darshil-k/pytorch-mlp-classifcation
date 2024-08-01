"""
This file is used to define the hyperparameters for the model.
"""
from typing import Optional
from pydantic import BaseModel

class HyperParameters(BaseModel):
    input_size: Optional[int] = 784 # default for MNIST
    num_classes: Optional[int] = 10 # default for MNIST
    num_color_channels: Optional[int] = 1 # default for MNIST
    num_epochs: Optional[int] = 5
    batch_size: Optional[int] = 100
    learning_rate: Optional[float] = 1e-3