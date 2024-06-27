""" Create dataset with all image files in a directory.
"""

import os
from pathlib import Path

import numpy as np
import skimage.io as io
import torch
from torch.utils.data import Dataset


def get_all_filenames(data_directory: str, extensions: list[str]):
    result = []
    for dirpath, dirnames, filenames in os.walk(data_directory):
        for filename in filenames:
            if Path(filename).suffix[1:] in extensions:
                result.append(os.path.join(dirpath, filename))
        for dirname in dirnames:
            result.extend(get_all_filenames(dirname, extensions))
    return result


class ImagesDataset(Dataset):
    """Create dataset with all image files in a directory."""

    def __init__(
        self,
        data_directory: str,
        extensions: list[str] = ["jpg"],
        transform=None,
    ):
        """Initializes the data set.

        Args:
            data_directory (str): The data directory.
            transform (torchvision.transforms, optional): A single or a set of
                transformations to modify the images. Defaults to None.
        """

        self.transform = transform
        self.filenames = sorted(get_all_filenames(data_directory, extensions))

    def __len__(self) -> int:
        """Return the number of items in the dataset.

        Returns:
            int: Number of items in dataset.
        """
        return len(self.filenames)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Retrieves the item/items with the given indices from the dataset.

        Args:
            index: The index of the item to retrieve.

        Returns:
            data: Data of the item/items with the given indices.

        """
        # Swap axis 0 and 2 to bring the color channel to the front
        data = io.imread(self.filenames[index])
        data = data.swapaxes(0, 2)
        data = torch.Tensor(data)
        if self.transform:
            data = self.transform(data)
        return data
