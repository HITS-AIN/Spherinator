import torch
import torchvision.transforms.v2 as transforms
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from spherinator.data.images_dataset import ImagesDataset


class ImagesDataModule(LightningDataModule):
    """Defines access to the ImagesDataset."""

    def __init__(
        self,
        data_directory: str,
        extensions: list[str] = ["jpg"],
        shuffle: bool = True,
        image_size: int = 64,
        batch_size: int = 32,
        num_workers: int = 1,
    ):
        """Initializes the data loader

        Args:
            data_directory (str): The data directory
            shuffle (bool, optional): Wether or not to shuffle whe reading. Defaults to True.
            image_size (int, optional): The size of the images. Defaults to 64.
            batch_size (int, optional): The batch size for training. Defaults to 32.
            num_workers (int, optional): How many worker to use for loading. Defaults to 1.
            download (bool, optional): Wether or not to download the data. Defaults to False.
        """
        super().__init__()

        self.data_directory = data_directory
        self.extensions = extensions
        self.shuffle = shuffle
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data_train = None
        self.dataloader_train = None

        self.transform_train = transforms.Compose(
            [
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

        if stage == "fit" and self.data_train is None:
            self.data_train = ImagesDataset(
                data_directory=self.data_directory,
                extensions=self.extensions,
                transform=self.transform_train,
            )
            self.dataloader_train = DataLoader(
                self.data_train,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
            )
        else:
            raise ValueError(f"Stage {stage} not supported.")

    def train_dataloader(self):
        """Gets the data loader for training."""
        return self.dataloader_train
