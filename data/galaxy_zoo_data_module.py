import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader

import data.preprocessing as preprocessing
from data.galaxy_zoo_dataset import GalaxyZooDataset

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

        self.train_transform = transforms.Compose(
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
        self.processing_transform = transforms.CenterCrop((363, 363))
        self.transform_thumbnail_images = transforms.Compose(
            [
                self.processing_transform,
                transforms.Resize((100, 100), antialias=True),
            ]
        )

    def setup(self, stage: str):
        """Sets up the data set and data loaders.

        Args:
            stage (str): Defines for which stage the data is needed.
        """
        if stage == "fit":
            self.data_train = GalaxyZooDataset(
                data_directory=self.data_dir,
                extension=self.extension,
                transform=self.train_transform,
            )
            self.dataloader_train = DataLoader(
                self.data_train,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
            )
        elif stage == "processing":
            self.data_val = GalaxyZooDataset(
                data_directory=self.data_dir,
                extension=self.extension,
                transform=self.processing_transform,
            )
            self.dataloader_val = DataLoader(
                self.data_train,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        elif stage == "images":
            self.data_val = GalaxyZooDataset(
                data_directory=self.data_dir,
                extension=self.extension,
                transform=self.images_transform,
            )
            self.dataloader_val = DataLoader(
                self.data_train,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        elif stage == "thumbnail_images":
            self.data_val = GalaxyZooDataset(
                data_directory=self.data_dir,
                extension=self.extension,
                transform=self.transform_thumbnail_transform,
            )
            self.dataloader_val = DataLoader(
                self.data_train,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        else:
            raise ValueError(f"Unknown stage: {stage}")
