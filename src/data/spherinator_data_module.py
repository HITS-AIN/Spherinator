from abc import ABC, abstractmethod
from pathlib import Path

import lightning.pytorch as pl

from models.spherinator_module import SpherinatorModule


class SpherinatorDataModule(ABC, pl.LightningDataModule):
    """
    Base class for all spherinator data modules.
    """

    def __init__(self):
        """Initialize SpherinatorDataModule."""
        super().__init__()

        self.data_train = None
        self.data_processing = None
        self.data_images = None
        self.data_thumbnail_images = None

        self.dataloader_train = None
        self.dataloader_processing = None
        self.dataloader_images = None
        self.dataloader_thumbnail_images = None

    def train_dataloader(self):
        """Gets the data loader for training."""
        return self.dataloader_train

    def processing_dataloader(self):
        """Gets the data loader for processing."""
        return self.dataloader_processing

    def images_dataloader(self):
        """Gets the data loader for images."""
        return self.dataloader_images

    def thumbnail_images_dataloader(self):
        """Gets the data loader for thumbnail images."""
        return self.dataloader_thumbnail_images

    @abstractmethod
    def write_catalog(
        self, model: SpherinatorModule, catalog_file: Path, hipster_url: str, title: str
    ):
        """Writes a catalog to disk."""

    def create_images(self, output_path: Path):
        """Writes images to disk."""
        raise NotImplementedError

    def create_thumbnails(self, output_path: Path):
        """Writes thumbnails to disk."""
        raise NotImplementedError
