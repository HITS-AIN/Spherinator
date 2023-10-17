""" Provides access to the Illustris sdss images.
"""
import os
from typing import List

import numpy
import torch
from astropy.io import fits
from torch.utils.data import Dataset


class IllustrisSdssDataset(Dataset):
    """ Provides access to Illustris sdss like images.
    """
    def __init__(self,
                 data_directories: List[str],
                 extension: str = ".fits",
                 minsize: int = 100,
                 transform = None):
        """ Initializes an Illustris sdss data set.

        Args:
            data_directories (List[str]): The directories to scan for images.
            extension (str, optional): The file extension to use for searching for files.
                Defaults to ".fits".
            transform (torchvision.transforms.Compose, optional): A single or a set of
                transformations to modify the images. Defaults to None.
        """
        self.data_directories = data_directories
        self.transform = transform
        self.files = []
        self.metadata = []
        self.total_files = 0
        for data_directory in data_directories:
            for file in os.listdir(data_directory):
                if file.endswith(extension):
                    self.total_files = self.total_files + 1
                    fits_filename = os.path.join(data_directory, file)
                    size = fits.getval(fits_filename, "NAXIS1")
                    if int(size) >= minsize:
                        self.files.append(fits_filename)
                        info = {}
                        splits = fits_filename[:-(len(extension))].split('/')
                        info['simulation'] = splits[-5]
                        info['snapshot'] = splits[-3].split('_')[1]
                        info['subhalo_id'] = splits[-1].split('_')[1]
                        self.metadata.append(info)
        self.len = len(self.files)

    def __len__(self):
        """ Return the number of items in the dataset.

        Returns:
            int: Number of items in dataset.
        """
        return self.len

    def __getitem__(self, idx):
        """ Retrieves the item/items with the given indices from the dataset.

        Args:
            idx (int or tensor): The index of the item to retrieve.

        Returns:
            dictionary: A dictionary mapping image, filename and id.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = fits.getdata(self.files[idx], 0)
        data = numpy.array(data).astype(numpy.float32)
        image = torch.Tensor(data)
        sample = {'image': image, 'filename': self.files[idx], 'id': idx, 'metadata': self.metadata[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample
