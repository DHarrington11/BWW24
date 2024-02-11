import os
import pickle

import torch
from torch.utils.data import Dataset

from .features import pipeline


class SatelliteDataset(Dataset):
    def __init__(self, data_dir, labels):
        """
        A PyTorch dataset for satellite images and their labels.

        Args:
        - data_dir (str): Path to directory containing the image data files.
        - labels (list): A list of labels for each image in the dataset.
        """
        self.data_dir = data_dir
        self.labels = labels

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
        - (int): The number of images in the dataset.
        """
        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
        - idx (int): The index of the item to retrieve.

        Returns:
        - image (torch.Tensor): The image data, loaded from a file.
        - label (torch.Tensor): The label for the image, converted to a float32 tensor.
        """
        label = torch.tensor(self.labels[idx])
        label = label.type(torch.float32)

        path = os.path.join(self.data_dir, f"{idx}.pickle")
        with open(path, "rb") as f:
            data = pickle.load(f)

        tensor = pipeline(data)
        return tensor, label
