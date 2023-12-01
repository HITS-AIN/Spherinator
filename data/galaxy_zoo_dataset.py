""" Provides access to the galaxy zoo images.
"""

import os

import numpy
import skimage.io as io
import torch

from .spherinator_dataset import SpherinatorDataset


class GalaxyZooDataset(SpherinatorDataset):
    """Provides access to galaxy zoo images."""

    def __init__(
        self,
        data_directory: str,
        extension: str = ".jpeg",
        label_file: str = str(),
        transform=None,
    ):
        """Initializes an galaxy zoo data set.

        Args:
            data_directory (str): The directory that contains the images for this dataset.
            extension (str, optional): The file extension to use when searching for file.
                Defaults to ".jpeg".
            label_file (str): The name of the file that contains the labels used for training of
                testing. By default None is specified. In this case no labels will be returned for
                the individual items!
            transform (torchvision.transforms.Compose, optional): A single or a set of
                transformations to modify the images. Defaults to None.
        """
        self.data_directory = data_directory
        self.transform = transform
        self.files = []
        for file in os.listdir(data_directory):
            if file.endswith(extension):
                self.files.append(os.path.join(data_directory, file))
        if label_file is str():
            self.labels = torch.Tensor(numpy.zeros(self.len))
        else:
            self.labels = torch.Tensor(
                numpy.loadtxt(label_file, delimiter=",", skiprows=1)[:, 1:]
            )

    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.files)

    def __getitem__(self, index: int):
        """Retrieves the item/items with the given indices from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            data: Data of the item/items with the given indices.
        """
        data = io.imread(self.files[index])
        data = torch.Tensor(data)
        # to normalize the RGB values to values between 0 and 1 ,swap 0,2 to get 3x424x424
        data = torch.swapaxes(data, 0, 2) / 255.0
        if self.transform:
            data = self.transform(data)
        return data

    def get_metadata(self, index: int):
        """Retrieves the metadata of the item/items with the given indices from the dataset.

        Args:
            index: The index of the item to retrieve.

        Returns:
            metadata: Metadata of the item/items with the given indices.
        """
        metadata = {
            "filename": self.files[index],
            "labels": self.labels[index],
        }
        return metadata
