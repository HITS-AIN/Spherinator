"""
This module is the entry point of the data package used to provide the access to the data.
It initializes the package and makes its modules available for import.

It contains the following modules:

1. `illustris_sdss_dataset`: Access to Illustris sdss data.
2. `illustris_sdss_data_module`: The illustris data module for training.
3. `galaxy_zoo_dataset`: Access to galaxy zoo data.
4. `galaxy_zoo_data_module`: The galaxy zoo data module for training.
5. `shapes_dataset`: Access to shapes data.
6. `shapes_data_module`: The shapes data module for training.
"""

from .illustris_sdss_dataset_multidim import IllustrisSdssDatasetMultidim
from .illustris_sdss_dataset import IllustrisSdssDataset
from .illustris_sdss_dataset_with_metadata import IllustrisSdssDatasetWithMetadata
from .illustris_sdss_data_module import IllustrisSdssDataModule
from .galaxy_zoo_dataset import GalaxyZooDataset
from .galaxy_zoo_data_module import GalaxyZooDataModule
from .shapes_dataset import ShapesDataset
from .shapes_data_module import ShapesDataModule

__all__ = [
    "IllustrisSdssDatasetMultidim",
    "IllustrisSdssDataset",
    "IllustrisSdssDatasetWithMetadata",
    "IllustrisSdssDataModule",
    "GalaxyZooDataset",
    "GalaxyZooDataModule",
    "ShapesDataset",
    "ShapesDataModule",
]
