""" Defines access to the ShapesDataset.
"""
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.shapes_dataset import ShapesDataset
from models.spherinator_module import SpherinatorModule

from .spherinator_data_module import SpherinatorDataModule


class ShapesDataModule(SpherinatorDataModule):
    """Defines access to the ShapesDataset."""

    def __init__(
        self,
        data_directory: str,
        exclude_files: Union[list[str], str] = [],
        shuffle: bool = True,
        image_size: int = 91,
        batch_size: int = 32,
        num_workers: int = 1,
        download: bool = False,
    ):
        """Initializes the data loader

        Args:
            data_directory (str): The data directory
            exclude_files (list[str] | str, optional): A list of files to exclude. Defaults to [].
            shuffle (bool, optional): Wether or not to shuffle whe reading. Defaults to True.
            image_size (int, optional): The size of the images. Defaults to 91.
            batch_size (int, optional): The batch size for training. Defaults to 32.
            num_workers (int, optional): How many worker to use for loading. Defaults to 1.
            download (bool, optional): Wether or not to download the data. Defaults to False.
        """
        super().__init__()

        self.data_directory = data_directory
        self.exclude_files = exclude_files
        self.shuffle = shuffle
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download

        self.transform_train = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Resize((self.image_size, self.image_size), antialias=True),
                transforms.Lambda(  # Normalize
                    lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x))
                ),
            ]
        )
        self.transform_processing = self.transform_train
        self.transform_images = self.transform_train
        self.transform_thumbnail_images = transforms.Compose(
            [
                self.transform_train,
                transforms.Resize((100, 100), antialias=True),
            ]
        )

    def setup(self, stage: str):
        """Sets up the data set and data loaders.

        Args:
            stage (str): Defines for which stage the data is needed.
                         For the moment just fitting is supported.
        """
        if not stage in ["fit", "processing", "images", "thumbnail_images"]:
            raise ValueError(f"Stage {stage} not supported.")

        if stage == "fit" and self.data_train is None:
            self.data_train = ShapesDataset(
                data_directory=self.data_directory,
                exclude_files=self.exclude_files,
                transform=self.transform_train,
                download=self.download,
            )
            self.dataloader_train = DataLoader(
                self.data_train,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
            )
        elif stage == "processing" and self.data_processing is None:
            self.data_processing = ShapesDataset(
                data_directory=self.data_directory,
                exclude_files=self.exclude_files,
                transform=self.transform_processing,
            )
            self.dataloader_processing = DataLoader(
                self.data_processing,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        elif stage == "images" and self.data_images is None:
            self.data_images = ShapesDataset(
                data_directory=self.data_directory,
                exclude_files=self.exclude_files,
                transform=self.transform_images,
            )
            self.dataloader_images = DataLoader(
                self.data_images,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
            )
        elif stage == "thumbnail_images" and self.data_thumbnail_images is None:
            self.data_thumbnail_images = ShapesDataset(
                data_directory=self.data_directory,
                exclude_files=self.exclude_files,
                transform=self.transform_thumbnail_images,
            )
            self.dataloader_thumbnail_images = DataLoader(
                self.data_thumbnail_images,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
            )

    def write_catalog(self, model: SpherinatorModule, catalog_file: Path):
        """Writes a catalog to disk."""
        self.setup("processing")
        with open(catalog_file, "w", encoding="utf-8") as output:
            output.write("#filename,RMSD,rotation,x,y,z\n")
