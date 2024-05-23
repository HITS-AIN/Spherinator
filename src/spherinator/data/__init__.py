"""
Entry point of the data package used to provide the access to the data.
"""

from .galaxy_zoo_data_module import GalaxyZooDataModule
from .galaxy_zoo_dataset import GalaxyZooDataset
from .illustris_sdss_data_module import IllustrisSdssDataModule
from .illustris_sdss_dataset import IllustrisSdssDataset
from .illustris_sdss_dataset_with_metadata import IllustrisSdssDatasetWithMetadata
from .images_data_module import ImagesDataModule
from .images_dataset import ImagesDataset
from .shapes_data_module import ShapesDataModule
from .shapes_dataset import ShapesDataset
from .spherinator_data_module import SpherinatorDataModule
from .spherinator_dataset import SpherinatorDataset

__all__ = [
    "GalaxyZooDataModule",
    "GalaxyZooDataset",
    "IllustrisSdssDataModule",
    "IllustrisSdssDataset",
    "IllustrisSdssDatasetWithMetadata",
    "ImagesDataModule",
    "ImagesDataset",
    "ShapesDataModule",
    "ShapesDataset",
    "SpherinatorDataModule",
    "SpherinatorDataset",
]
