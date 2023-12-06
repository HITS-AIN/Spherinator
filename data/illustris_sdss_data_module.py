import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader

import data.preprocessing as preprocessing
from data.illustris_sdss_dataset import IllustrisSdssDataset

from .spherinator_data_module import SpherinatorDataModule


class IllustrisSdssDataModule(SpherinatorDataModule):
    """Defines access to the Illustris sdss data as a data module."""

    def __init__(
        self,
        data_directories: list[str],
        extension: str = "fits",
        minsize: int = 100,
        shuffle: bool = True,
        batch_size: int = 32,
        num_workers: int = 16,
    ):
        """Initialize IllustrisSdssDataModule.

        Args:
            data_directories (list[str]): The directories to scan for data files.
            extension (str, optional): The kind of files to search for. Defaults to "fits".
            minsize (int, optional): The minimum size a file should have. Defaults to 100 pixels.
            shuffle (bool, optional): Wether or not to shuffle whe reading. Defaults to True.
            batch_size (int, optional): The batch size for training. Defaults to 32.
            num_workers (int, optional): How many worker to use for loading. Defaults to 16.
        """
        super().__init__()

        self.data_directories = data_directories
        self.extension = extension
        self.minsize = minsize
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform_images = transforms.Compose(
            [
                preprocessing.CreateNormalizedRGBColors(
                    stretch=0.9,
                    range=5,
                    lower_limit=0.001,
                    channel_combinations=[[2, 3], [1, 0], [0]],
                    scalers=[0.7, 0.5, 1.3],
                ),
            ]
        )
        self.transform_processing = transforms.Compose(
            [
                transforms.CenterCrop((363, 363)),
                self.transform_images,
            ]
        )
        self.transform_train = transforms.Compose(
            [
                self.transform_processing,
                preprocessing.DielemanTransformation(
                    rotation_range=[0, 360],
                    translation_range=[0, 0],  # 4./363,4./363],
                    scaling_range=[1, 1],  # 0.9,1.1],
                    flip=0.5,
                ),
                transforms.CenterCrop((363, 363)),
            ]
        )
        self.transform_thumbnail_images = transforms.Compose(
            [
                transforms.CenterCrop((363, 363)),
                transforms.Resize((100, 100), antialias=True),
                self.transform_images,
            ]
        )

    def setup(self, stage: str):
        """Sets up the data set and data loaders.

        Args:
            stage (str): Defines for which stage the data is needed.
        """
        if stage == "fit":
            self.data_train = IllustrisSdssDataset(
                data_directories=self.data_directories,
                extension=self.extension,
                minsize=self.minsize,
                transform=self.transform_train,
            )

            self.dataloader_train = DataLoader(
                self.data_train,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
            )
        elif stage == "processing":
            self.data_processing = IllustrisSdssDataset(
                data_directories=self.data_directories,
                extension=self.extension,
                minsize=self.minsize,
                transform=self.transform_processing,
            )

            self.dataloader_processing = DataLoader(
                self.data_processing,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        elif stage == "images":
            self.data_images = IllustrisSdssDataset(
                data_directories=self.data_directories,
                extension=self.extension,
                minsize=self.minsize,
                transform=self.transform_images,
            )

            self.dataloader_images = DataLoader(
                self.data_images,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
            )
        elif stage == "thumbnail_images":
            self.data_thumbnail_images = IllustrisSdssDataset(
                data_directories=self.data_directories,
                extension=self.extension,
                minsize=self.minsize,
                transform=self.transform_thumbnail_images,
            )

            self.dataloader_thumbnail_images = DataLoader(
                self.data_thumbnail_images,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
            )
        else:
            raise ValueError(f"Unknown stage: {stage}")
