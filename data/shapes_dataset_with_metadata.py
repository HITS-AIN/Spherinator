import numpy

from .shapes_dataset import ShapesDataset


class ShapesDatasetWithMetadata(ShapesDataset):
    """Provides access to shapes images and metadata."""

    def __getitem__(self, index: int):
        """Retrieves the item/items with the given indices from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            data: Data of the item/items with the given indices.
            metadata: Metadata of the item/items with the given indices.
        """
        data = super().__getitem__(index)

        file_index = (
            numpy.searchsorted(self.file_entries_offsets, index, side="right") - 1
        )
        metadata = {
            "filename": self.filenames[file_index],
        }

        return data, metadata
