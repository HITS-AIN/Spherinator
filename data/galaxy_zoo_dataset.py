""" Provides access to the galaxy zoo images.
"""

import os
import numpy
import torch
from torch.utils.data import Dataset
import skimage.io as io

class GalaxyZooDataset(Dataset):
    """ Provides access to galaxy zoo images.
    """
    def __init__(self,
                 data_directory: str,
                 extension: str = ".jpeg",
                 label_file = None,
                 transform = None):
        """ Initializes an galaxy zoo data set.

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
        self.len = len(self.files)
        if label_file is None:
            self.labels = torch.Tensor(numpy.zeros(self.len))
        else:
            self.labels = torch.Tensor(numpy.loadtxt(label_file, delimiter=',', skiprows=1)[:,1:])

    def __len__(self):
        """ Return the number of items in the dataset.

        Returns:
            int: Number of items in dataset.
        """
        return self.len

    def __getitem__(self, idx):
        """ Retrieves the item/items with the given indices from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dictionary: A dictionary mapping image, filename, labels and id.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # to normalize the RGB values to values between 0 and 1 ,swap 0,2 to get 3x424x424
        image = torch.swapaxes(torch.Tensor(io.imread(self.files[idx])), 0, 2) / 255.0
        sample = {'image': image,
                  'filename': self.files[idx],
                  'labels': self.labels[idx],
                  'id': idx}
        if self.transform:
            sample = self.transform(sample)
        return sample
