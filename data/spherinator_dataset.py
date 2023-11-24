from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class SpherinatorDataset(ABC, Dataset):
    """
    Abstract base class for all datasets
    """

    @abstractmethod
    def get_metadata(self):
        """Returns the metadata of the dataset."""
