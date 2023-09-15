""" Provides access to the Illustris sdss data set.
"""
from typing import List

import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchvision import transforms

import data.preprocessing as preprocessing
from data.illustris_sdss_dataset import IllustrisSdssDataset

class IllustrisSdssDataModule(pl.LightningDataModule):
    """ Defines access to the Illustris sdss data as a data module.
    """
    def __init__(self,
                 data_directories: List[str],
                 extension: str = "fits",
                 minsize: int = 100,
                 shuffle: bool = True,
                 batch_size: int = 32,
                 num_workers: int = 16):
        """ Initializes the data loader for the Illustris sdss images.

        Args:
            data_directories (List[str]): The directories to scan for data files.
            extension (str, optional): The kind of files to search for. Defaults to "fits".
            minsize (int, optional): The minimum size a file should have. Defaults to 100 pixels.
            shuffle (bool, optional): Wether or not to shuffle whe reading. Defaults to True.
            batch_size (int, optional): The batch size for training. Defaults to 32.
            num_workers (int, optional): How many worker to use for loading. Defaults to 16.
        """
        super().__init__()

        self.data_directories = data_directories
        self.transform_train = transforms.Compose([
            preprocessing.CropAndExpand((363, 363)),
            preprocessing.CreateNormalizedColors(stretch=0.9,
                                                 range=5,
                                                 lower_limit=0.001,
                                                 channel_combinations=[[2, 3],[1,0],[0]],
                                                 scalers=[.7, .5, 1.3]),
            preprocessing.DielemanTransformation(
                rotation_range=[0,360],
                translation_range=[0,0],#4./363,4./363],
                scaling_range=[1,1],#0.9,1.1],
                flip=0.5),
            preprocessing.CropAndScale((363,363), (363,363))
        ])

        self.transform_predict = transforms.Compose([
            preprocessing.CropAndExpand((363, 363)),
            preprocessing.CreateNormalizedColors(stretch=0.9,
                                                 range=5,
                                                 lower_limit=0.001,
                                                 channel_combinations=[[2, 3],[1,0],[0]],
                                                 scalers=[.7, .5, 1.3]),
        ])

        self.transform_val = transforms.Compose([
            preprocessing.CropAndScale((363,363), (100,100)),
            preprocessing.CreateNormalizedColors(stretch=0.9,
                                                 range=5,
                                                 lower_limit=0.001,
                                                 channel_combinations=[[2, 3],[1,0],[0]],
                                                 scalers=[.7, .5, 1.3]),
        ])

        self.batch_size = batch_size
        self.extension = extension
        self.minsize = minsize
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.data_train = None
        self.dataloader_train = None
        self.data_predict = None
        self.dataloader_predict = None
        self.data_val = None
        self.dataloader_val = None

    def setup(self, stage: str):
        """ Sets up the data set and data loaders.

        Args:
            stage (str): Defines for which stage the data is needed. For the moment just fitting
                is supported.
        """
        if stage == "fit":
            self.data_train = IllustrisSdssDataset(data_directories=self.data_directories,
                                                   extension=self.extension,
                                                   minsize=self.minsize,
                                                   transform=self.transform_train)

            self.dataloader_train = DataLoader(self.data_train,
                                              batch_size=self.batch_size,
                                              shuffle=self.shuffle,
                                              num_workers=self.num_workers)
        if stage == "predict":
            self.data_predict = IllustrisSdssDataset(data_directories=self.data_directories,
                                                     extension=self.extension,
                                                     minsize=self.minsize,
                                                     transform=self.transform_predict)

            self.dataloader_predict = DataLoader(self.data_predict,
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 num_workers=self.num_workers)

        if stage == "val":
            self.data_val = IllustrisSdssDataset(data_directories=self.data_directories,
                                                 extension=self.extension,
                                                 minsize=self.minsize,
                                                 transform=self.transform_val)

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
