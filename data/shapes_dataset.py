""" Test images with four shapes in random rotations.
"""
from typing import List

import os
import numpy as np

import torch
from torch.utils.data import Dataset


class ShapesDataset(Dataset):
    """Test images with four shapes in random rotations."""

    def __init__(self, data_directory: str, transform=None, download: bool = False):
        """Initializes an Illustris sdss data set.

        Args:
            data_directory (str): The data directory.
            transform (torchvision.transforms.Compose, optional): A single or a set of
                transformations to modify the images. Defaults to None.
            download (bool, optional): Wether or not to download the data. Defaults to False.
        """
        self.data_directory = data_directory
        self.transform = transform
        self.download = download

        if self.download:
            raise NotImplementedError("Download not implemented yet.")

        self.images = np.empty((0, 64, 64), np.float32)
        for file in os.listdir(data_directory):
            images = np.load(os.path.join(data_directory, file)).astype(np.float32)
            # for i in range(images.shape[0]):
            #     images[i, :, :] = (images[i, :, :] - np.min(images[i, :, :])) / (
            #         np.max(images[i, :, :]) - np.min(images[i, :, :])
            #     )  # Normalize
            self.images = np.append(self.images, images, axis=0)

    def __len__(self):
        """Return the number of items in the dataset.

        Returns:
            int: Number of items in dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """Retrieves the item/items with the given indices from the dataset.

        Args:
            idx (int or tensor): The index of the item to retrieve.

        Returns:
            dictionary: A dictionary mapping image, filename and id.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = torch.Tensor(self.images[idx])
        if self.transform:
            image = self.transform(image)
        sample = {"image": image, "id": idx}
        return sample
