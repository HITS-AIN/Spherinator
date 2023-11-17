""" Test images with four shapes in random rotations.
"""
import os
from typing import List

import numpy as np
from sklearn import dummy
import torch
from torch.utils.data import Dataset


class ShapesDataset(Dataset):
    """Test images with four shapes in random rotations."""

    def __init__(
        self,
        data_directory: str,
        exclude_files: List[str] | str = [],
        transform=None,
        download: bool = False,
    ):
        """Initializes the data set.

        Args:
            data_directory (str): The data directory.
            exclude_files (List[str], optional): A list of files to exclude. Defaults to [].
            transform (torchvision.transforms.Compose, optional): A single or a set of
                transformations to modify the images. Defaults to None.
            download (bool, optional): Wether or not to download the data. Defaults to False.
        """

        if isinstance(exclude_files, str):
            exclude_files = [exclude_files]

        self.data_directory = data_directory
        self.exclude_files = exclude_files
        self.transform = transform
        self.download = download

        if self.download:
            raise NotImplementedError("Download not implemented yet.")

        self.images = np.empty((0, 64, 64), np.float32)
        for file in os.listdir(data_directory):
            if file in exclude_files:
                continue
            images = np.load(os.path.join(data_directory, file)).astype(np.float32)
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
        dummy_filename = "dummy"
        dummy_metadata = {"simulation": "shapes", "snapshot": "0", "subhalo_id": "0"}
        sample = {'image': image, 'filename': dummy_filename, 'id': idx, 'metadata': dummy_metadata}
        return sample
