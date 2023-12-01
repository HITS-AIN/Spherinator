import lightning.pytorch as pl


class SpherinatorDataModule(pl.LightningDataModule):
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
