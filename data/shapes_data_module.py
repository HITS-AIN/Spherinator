""" Defines access to the ShapesDataset.
"""
from typing import List

import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from data.shapes_dataset import ShapesDataset

class ShapesDataModule(pl.LightningDataModule):
    """ Defines access to the ShapesDataset.
    """
    def __init__(self,
                 data_directory: str,
                 shuffle: bool = True,
                 batch_size: int = 32,
                 num_workers: int = 1):
        """ Initializes the data loader

        Args:
            data_directories (List[str]): The data directory
            shuffle (bool, optional): Wether or not to shuffle whe reading. Defaults to True.
            batch_size (int, optional): The batch size for training. Defaults to 32.
            num_workers (int, optional): How many worker to use for loading. Defaults to 1.
        """
        super().__init__()

        self.data_directory = data_directory
        self.transform_train = transforms.Compose([
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0,0,0), (290,290,290)),
            transforms.Resize((363,363))
        ])

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.data_train = None
        self.dataloader_train = None

    def setup(self, stage: str):
        """ Sets up the data set and data loaders.

        Args:
            stage (str): Defines for which stage the data is needed.
                         For the moment just fitting is supported.
        """
        if stage == "fit":
            self.data_train = ShapesDataset(data_directory=self.data_directory,
                                            transform=self.transform_train)

            self.dataloader_train = DataLoader(self.data_train,
                                              batch_size=self.batch_size,
                                              shuffle=self.shuffle,
                                              num_workers=self.num_workers)

    def train_dataloader(self):
        """ Gets the data loader for training.

        Returns:
            torch.utils.data.DataLoader: The dataloader instance to use for training.
        """
        return self.dataloader_train
