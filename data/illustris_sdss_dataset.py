""" Provides access to the Illustris sdss images.
"""
import os
import torch

import numpy
from astropy.io import fits

from .spherinator_dataset import SpherinatorDataset


class IllustrisSdssDataset(SpherinatorDataset):
    """Provides access to Illustris sdss images."""

    def __init__(
        self,
        data_directories: list[str],
        extension: str = ".fits",
        minsize: int = 100,
        transform=None,
    ):
        """Initializes an Illustris sdss data set.

        Args:
            data_directories (list[str]): The directories to scan for images.
            extension (str, optional): The file extension to use for searching for files.
                Defaults to ".fits".
            minsize (int, optional): The minimum size of the images to include. Defaults to 100.
            transform (torchvision.transforms.Compose, optional): A single or a set of
                transformations to modify the images. Defaults to None.
        """
        self.data_directories = data_directories
        self.extension = extension
        self.transform = transform
        self.files = []
        self.total_files = 0
        for data_directory in data_directories:
            for file in sorted(os.listdir(data_directory)):
                if file.endswith(extension):
                    self.total_files = self.total_files + 1
                    fits_filename = os.path.join(data_directory, file)
                    size = fits.getval(fits_filename, "NAXIS1")
                    if int(size) >= minsize:
                        self.files.append(fits_filename)

    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.files)

    def __getitem__(self, index: int):
        """Retrieves the item/items with the given indices from the dataset.

        Args:
            index: The index of the item to retrieve.

        Returns:
            data: Data of the item/items with the given indices.
        """
        data = fits.getdata(self.files[index], 0)
        data = numpy.array(data).astype(numpy.float32)
        data = torch.Tensor(data)
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
        filename = self.files[index]
        splits = filename[: -(len(self.extension) + 1)].split("/")
        metadata = {
            "filename": self.files[index],
            "simulation": splits[-5],
            "snapshot": splits[-3].split("_")[1],
            "subhalo_id": splits[-1].split("_")[1],
        }
        return metadata
