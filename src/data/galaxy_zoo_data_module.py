from pathlib import Path

import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader

import data.preprocessing as preprocessing
from data.galaxy_zoo_dataset import GalaxyZooDataset
from models.spherinator_module import SpherinatorModule

from .spherinator_data_module import SpherinatorDataModule


class GalaxyZooDataModule(SpherinatorDataModule):
    """Defines access to the Galaxy Zoo data as a data module."""

    def __init__(
        self,
        data_directory: str = "./",
        batch_size: int = 32,
        extension: str = "jpg",
        shuffle: bool = True,
        num_workers: int = 16,
    ):
        """Initialize GalaxyZooDataModule

        Args:
            data_directory (str): The directories to scan for data files.
            batch_size (int, optional): The batch size for training. Defaults to 32.
            extension (str, optional): The kind of files to search for. Defaults to "jpg".
            shuffle (bool, optional): Wether or not to shuffle whe reading. Defaults to True.
            num_workers (int, optional): How many worker to use for loading. Defaults to 16.
        """
        super().__init__()

        self.data_directory = data_directory
        self.batch_size = batch_size
        self.extension = extension
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.transform_train = transforms.Compose(
            [
                preprocessing.DielemanTransformation(
                    rotation_range=[0, 360],
                    translation_range=[4.0 / 424, 4.0 / 424],
                    scaling_range=[1 / 1.1, 1.1],
                    flip=0.5,
                ),
                transforms.CenterCrop((363, 363)),
                transforms.Resize((424, 424), antialias=True),
            ]
        )
        self.transform_processing = transforms.CenterCrop((363, 363))
        self.transform_images = self.transform_train
        self.transform_thumbnail_images = transforms.Compose(
            [
                self.transform_processing,
                transforms.Resize((100, 100), antialias=True),
            ]
        )

    def setup(self, stage: str):
        """Sets up the data set and data loaders.

        Args:
            stage (str): Defines for which stage the data is needed.
        """
        if not stage in ["fit", "processing", "images", "thumbnail_images"]:
            raise ValueError(f"Stage {stage} not supported.")

        if stage == "fit" and self.data_train is None:
            self.data_train = GalaxyZooDataset(
                data_directory=self.data_directory,
                extension=self.extension,
                transform=self.transform_train,
            )
            self.dataloader_train = DataLoader(
                self.data_train,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
            )
        elif stage == "processing" and self.data_processing is None:
            self.data_processing = GalaxyZooDataset(
                data_directory=self.data_directory,
                extension=self.extension,
                transform=self.transform_processing,
            )
            self.dataloader_processing = DataLoader(
                self.data_train,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        elif stage == "images" and self.data_images is None:
            self.data_images = GalaxyZooDataset(
                data_directory=self.data_directory,
                extension=self.extension,
                transform=self.transform_images,
            )
            self.dataloader_images = DataLoader(
                self.data_train,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        elif stage == "thumbnail_images" and self.data_thumbnail_images is None:
            self.data_thumbnail_images = GalaxyZooDataset(
                data_directory=self.data_directory,
                extension=self.extension,
                transform=self.transform_thumbnail_images,
            )
            self.dataloader_thumbnail_images = DataLoader(
                self.data_train,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

    def write_catalog(
        self, model: SpherinatorModule, catalog_file: Path, hipster_url: str, title: str
    ):
        """Writes a catalog to disk."""
        self.setup("processing")
        with open(catalog_file, "w", encoding="utf-8") as output:
            output.write("#filename,RMSD,rotation,x,y,z\n")
