""" Test images with four shapes in random rotations.
"""
import os
from typing import Union

import numpy as np
import torch

from .spherinator_dataset import SpherinatorDataset


class ShapesDataset(SpherinatorDataset):
    """Test images with four shapes in random rotations."""

    def __init__(
        self,
        data_directory: str,
        exclude_files: Union[list[str], str] = [],
        transform=None,
        download: bool = False,
    ):
        """Initializes the data set.

        Args:
            data_directory (str): The data directory.
            exclude_files (list[str] | str, optional): A list of files to exclude. Defaults to [].
            transform (torchvision.transforms, optional): A single or a set of
                transformations to modify the images. Defaults to None.
            download (bool, optional): Wether or not to download the data. Defaults to False.
        """

        if isinstance(exclude_files, str):
            exclude_files = [exclude_files]

        self.transform = transform
        self.filenames = []

        if download:
            raise NotImplementedError("Download not implemented yet.")

        self.images = np.empty((0, 64, 64), np.float32)
        for file in os.listdir(data_directory):
            if file in exclude_files:
                continue
            self.filenames.append(os.path.join(data_directory, file))
            images = np.load(os.path.join(data_directory, file)).astype(np.float32)
            self.images = np.append(self.images, images, axis=0)

    def __len__(self):
        """Return the number of items in the dataset.

        Returns:
            int: Number of items in dataset.
        """
        return len(self.images)

    def __getitem__(self, index):
        """Retrieves the item/items with the given indices from the dataset.

        Args:
            index: The index of the item to retrieve.

        Returns:
            data: Data of the item/items with the given indices.
            index: Index of the item/items

        """
        if torch.is_tensor(index):
            index = index.tolist()
        data = torch.Tensor(self.images[index])
        if self.transform:
            data = self.transform(data)
        return data, index

    def get_metadata(self, index):
        """Retrieves the metadata of the item/items with the given indices from the dataset.

        Args:
            index: The index of the item to retrieve.

        Returns:
            metadata: Metadata of the item/items with the given indices.
        """
        if torch.is_tensor(index):
            index = index.tolist()
        metadata = {
            "index": index,
            "filename": self.filenames[index],
        }
        return metadata
