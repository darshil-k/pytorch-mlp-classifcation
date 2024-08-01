"""
This file is used to prepare the train and test data for the model.
It prepares the MNIST dataset for the model.
"""
from typing import Literal

# Importing the required libraries
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

class MNISTDataPreparation:
    """
    This class is used to prepare the train and test data for the model.
    If the data is not present in the data folder, it downloads the data.
    """
    def __init__(self, batch_size: int, data_dir: str, is_download: bool):
        """
        This method initializes the class.
        :param batch_size: The batch size for the data.
        :param data_dir: Path of directory where data is present or should be downloaded to.
        :param is_download: If True, downloads the data, else uses the data present in the directory.
        """
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.is_download = is_download
        # placeholder for data
        self.train_data = []
        self.test_data = []

        if not is_download:
            assert os.path.exists(data_dir), "Data directory does not exist. Please set is_download to True to download the data."
        else:
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

    def transform_image_flatten(self):
        """
        This method defined & returns transforms object of torchvision.transforms.Compose
        :return: The transform object.
        """
        # flatten the image and convert to tensor
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
        return transform


    def prepare_data(self) -> (DataLoader, DataLoader):
        """
        This method prepares the train and test data for the model.
        :return: The train and test data loaders.
        """
        # Loading the train and test data
        self.train_data = datasets.MNIST(root=self.data_dir, train=True, download=self.is_download, transform=self.transform_image_flatten())
        self.test_data = datasets.MNIST(root=self.data_dir, train=False, download=self.is_download, transform=self.transform_image_flatten())

        # Creating the data loaders
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True)

        return train_loader, test_loader

    def get_length(self, type: Literal["train", "test"]) -> int:
        """
        This method returns the length of the dataset.
        :param type: The type of dataset. It can be train or test.
        :return: The length of the dataset.
        """
        if type == "train":
            length = len(self.train_data)
        elif type == "test":
            length = len(self.test_data)
        else:
            raise ValueError("Invalid type. Please provide either train or test.")
        return length

    def get_steps_per_epoch(self) -> int:
        """
        This method returns the steps per epoch.
        :return: The steps per epoch.
        """
        steps_per_epoch = np.ceil(self.get_length("train") / self.batch_size)
        return int(steps_per_epoch)

