"""
This file is used to prepare the train and test data for the model.
It prepares the MNIST dataset for the model.
"""
from typing import Literal

# Importing the required libraries
import numpy as np
import torch
from pydantic import BaseModel
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd

class MNISTDataClasses(BaseModel):
    """
    This is a mapping of MNIST data classes to their respective indexes.
    """
    digit_zero: int = 0
    digit_one: int = 1
    digit_two: int = 2
    digit_three: int = 3
    digit_four: int = 4
    digit_five: int = 5
    digit_six: int = 6
    digit_seven: int = 7
    digit_eight: int = 8
    digit_nine: int = 9

    def get_class_name_mapping(self) -> dict:
        """
        This method returns the class name mapping.
        :return: The class name mapping.
        """
        return self.__dict__

    def get_class_name(self, class_index: int) -> str:
        """
        This method returns the class name for the given class index.
        :param class_index: The class index.
        :return: The class name.
        """
        class_name = ""
        for key, value in self.get_class_name_mapping().items():
            if value == class_index:
                class_name = key
        return class_name

    def get_class_names(self) -> list[str]:
        """
        This method returns the list of class names in order of index.
        :return: The list of class names.
        """
        class_names = []
        sort_classes = sorted(self.classes.__dict__.items(), key=lambda x: x[1])
        for key, value in sort_classes:
            class_names.append(key)
        return class_names


class MNISTCustomDataset(Dataset):
    """
    This class is used to create a custom dataset for MNIST.
    The data is loaded from disk.
    """
    def __init__(self, data_dir: str, train: bool):
        """
        This method initializes the class.
        :param data_dir: The directory where the data (train.csv & test.csv) is present.
        :param train: If True, loads the train data, else loads the test data.
        """
        assert os.path.exists(data_dir), "Data directory does not exist. Please set is_download to True to download the data."
        assert os.path.exists(os.path.join(data_dir, "mnist_train.csv")), "Train data file does not exist."
        assert os.path.exists(os.path.join(data_dir, "mnist_test.csv")), "Test data file does not exist."
        self.data_dir = data_dir
        self.is_train = train

        if train:
            self.dataframe = pd.read_csv(os.path.join(data_dir, "mnist_train.csv"))
        else:
            self.dataframe = pd.read_csv(os.path.join(data_dir, "mnist_test.csv"))

    def __len__(self):
        """
        This method returns the length of the dataset.
        :return:
        """
        return len(self.dataframe)

    def __getitem__(self, idx) -> (np.ndarray, int):
        """
        This method returns the image and label for the given index.
        :param idx: index of the image
        :return: Flatten numpy array of image and label as int
        """
        # label is the first column
        label = self.dataframe.iloc[idx, 0]

        # image (flatten data) is the rest of the columns
        image = self.dataframe.iloc[idx, 1:].values
        image = image.astype(np.float32)
        return image, label



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
        self.classes = MNISTDataClasses()
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
        ## Following two lines should be used to load data if it is available as official format. If the data is not present than use download=True
        # self.train_data = datasets.MNIST(root=self.data_dir, train=True, download=self.is_download, transform=self.transform_image_flatten())
        # self.test_data = datasets.MNIST(root=self.data_dir, train=False, download=self.is_download, transform=self.transform_image_flatten())

        ## Use custom dataset to load data from disk
        self.train_data = MNISTCustomDataset(data_dir=self.data_dir, train=True)
        self.test_data = MNISTCustomDataset(data_dir=self.data_dir, train=False)

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

    def get_classes(self) -> MNISTDataClasses:
        """
        This method returns the MNIST data classes.
        :return: The MNIST data classes.
        """
        return self.classes


