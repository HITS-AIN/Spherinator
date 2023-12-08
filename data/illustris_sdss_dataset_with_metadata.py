import numpy
import torch
from astropy.io import fits

from .illustris_sdss_dataset import IllustrisSdssDataset


class IllustrisSdssDatasetWithMetadata(IllustrisSdssDataset):
    """Provides access to Illustris sdss images and metadata."""

    def __getitem__(self, index: int):
        """Retrieves the item/items with the given indices from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            data: Data of the item/items with the given indices.
            metadata: Metadata of the item/items with the given indices.
        """
        data = super().__getitem__(index)

        filename = self.files[index]
        splits = filename[: -(len(self.extension) + 1)].split("/")
        metadata = {
            "filename": filename,
            "simulation": splits[-5],
            "snapshot": splits[-3].split("_")[1],
            "subhalo_id": splits[-1].split("_")[1],
        }

        return data, metadata
