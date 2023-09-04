"""
This module is the entry point of the data package used to provide the access to the data.
It initializes the package and makes its modules available for import.

It contains the following modules:

1. `illustris_sdss_dataset`: Access to Illustris sdss data.
2. `illustris_sdss_data_module`: The illustris data module for training.
3. `galaxy_zoo_dataset`: Access to galaxy zoo data.
4. `galaxy_zoo_data_module`: The galaxy zoo data module for training.
"""

from .illustris_sdss_dataset import IllustrisSdssDataset
from .illustris_sdss_data_module import IllustrisSdssDataModule
from .galaxy_zoo_dataset import GalaxyZooDataset
from .galaxy_zoo_data_module import GalaxyZooDataModule

__all__ = [
    'IllustrisSdssDataset',
    'IllustrisSdssDataModule',
    'GalaxyZooDataset',
    'GalaxyZooDataModule'
]
