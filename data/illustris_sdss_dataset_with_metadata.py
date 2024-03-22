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
        return data, self.get_metadata(index)
