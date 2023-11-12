""" Defines access to the ShapesDataset.
"""
from typing import List

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.shapes_dataset import ShapesDataset


class ShapesDataModule(pl.LightningDataModule):
    """ Defines access to the ShapesDataset.
    """
    def __init__(self,
                 data_directory: str,
                 shuffle: bool = True,
                 image_size: int = 91,
                 batch_size: int = 32,
                 num_workers: int = 1,
                 download: bool = False):
        """ Initializes the data loader

        Args:
            data_directories (List[str]): The data directory
            shuffle (bool, optional): Wether or not to shuffle whe reading. Defaults to True.
            image_size (int, optional): The size of the images. Defaults to 91.
            batch_size (int, optional): The batch size for training. Defaults to 32.
            num_workers (int, optional): How many worker to use for loading. Defaults to 1.
            download (bool, optional): Wether or not to download the data. Defaults to False.
        """
        super().__init__()

        self.data_directory = data_directory
        self.shuffle = shuffle
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download

        self.transform_train = transforms.Compose([
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Resize((self.image_size, self.image_size), antialias="none"),
            transforms.Lambda(lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x))),
        ])
        self.transform_predict = self.transform_train
        self.transform_val = self.transform_train

        self.data_train = None
        self.dataloader_train = None
        self.data_predict = None
        self.dataloader_predict = None
        self.data_val = None
        self.dataloader_val = None

    def setup(self, stage: str):
        """ Sets up the data set and data loaders.

        Args:
            stage (str): Defines for which stage the data is needed.
                         For the moment just fitting is supported.
        """
        if stage == "fit":
            self.data_train = ShapesDataset(data_directory=self.data_directory,
                                            transform=self.transform_train,
                                            download=self.download)

            self.dataloader_train = DataLoader(self.data_train,
                                              batch_size=self.batch_size,
                                              shuffle=self.shuffle,
                                              num_workers=self.num_workers)
        if stage == "predict":
            self.data_predict = ShapesDataset(data_directory=self.data_directory,
                                              transform=self.transform_predict,
                                              download=self.download)

            self.dataloader_predict = DataLoader(self.data_predict,
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 num_workers=self.num_workers)

        if stage == "val":
            self.data_val = ShapesDataset(data_directory=self.data_directory,
                                          transform=self.transform_val,
                                          download=self.download)

            self.dataloader_val = DataLoader(self.data_val,
                                             batch_size=self.batch_size,
                                             shuffle=False,
                                             num_workers=self.num_workers)

    def train_dataloader(self):
        """ Gets the data loader for training.

        Returns:
            torch.utils.data.DataLoader: The dataloader instance to use for training.
        """
        return self.dataloader_train

    def predict_dataloader(self):
        """ Gets the data loader for prediction.

        Returns:
            torch.utils.data.DataLoader: The dataloader instance to use for prediction.
        """
        return self.dataloader_predict

    def val_dataloader(self):
        """ Gets the data loader for validation.

        Returns:
            torch.utils.data.DataLoader: The dataloader instance to use for validation.
        """
        return self.dataloader_val
