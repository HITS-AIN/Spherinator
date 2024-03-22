import math

import healpy
import numpy
import torch
from PIL import Image
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from models.spherinator_module import SpherinatorModule
from .illustris_sdss_dataset import IllustrisSdssDataset
from .illustris_sdss_data_module import IllustrisSdssDataModule
from .illustris_sdss_dataset_multidim import IllustrisSdssDatasetMultidim
import open3d as o3d


class IllustrisSdssDataModuleMultidim(IllustrisSdssDataModule):
    def __init__(
            self,
            data_directories: list[str],
            cutout_directory: str,
            info_directory: str,
            extension: str = "fits",
            minsize: int = 100,
            shuffle: bool = True,
            batch_size: int = 32,
            num_workers: int = 16,
    ):
        super().__init__(data_directories, extension, minsize, shuffle, batch_size, num_workers)
        self.cutout_directory = cutout_directory
        self.info_directory = info_directory

    def setup(self, stage: str):
        if not stage in ["fit", "processing", "images", "thumbnail_images", "gas_pointclouds"]:
            raise ValueError(f"Stage {stage} not supported.")

        if stage == "fit" and self.data_train is None:
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
        elif stage == "processing" and self.data_processing is None:
            self.data_processing = IllustrisSdssDatasetMultidim(
                data_directories=self.data_directories,
                cutout_directory=self.cutout_directory,
                info_dir=self.info_directory,
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
        elif stage == "images" and self.data_images is None:
            self.data_images = IllustrisSdssDatasetMultidim(
                data_directories=self.data_directories,
                cutout_directory=self.cutout_directory,
                info_dir=self.info_directory,
                extension=self.extension,
                minsize=self.minsize,
                transform=self.transform_processing,
            )
            self.dataloader_images = DataLoader(
                self.data_images,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
            )
        elif stage == "thumbnail_images" and self.data_thumbnail_images is None:
            self.data_thumbnail_images = IllustrisSdssDatasetMultidim(
                data_directories=self.data_directories,
                cutout_directory=self.cutout_directory,
                info_dir=self.info_directory,
                extension=self.extension,
                minsize=self.minsize,
                transform=self.transform_processing,
            )
            self.dataloader_thumbnail_images = DataLoader(
                self.data_thumbnail_images,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
            )
        if stage == "gas_pointclouds":
            self.data_gas_pointclouds = IllustrisSdssDatasetMultidim(
                data_directories=self.data_directories,
                cutout_directory=self.cutout_directory,
                info_dir=self.info_directory,
                data_aspect="gas_pointcloud",
                extension=self.extension,
                minsize=self.minsize,
                transform=self.transform_processing,
            )
            self.dataloader_gas_pointclouds = DataLoader(
                self.data_gas_pointclouds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

    def write_catalog(
        self, model: SpherinatorModule, catalog_file: Path, hipster_url: str, title: str
    ):
        raise NotImplementedError("This function does not exist for this data module.")

    def create_images(self, output_path: Path):
        """Writes preview images to disk.

        Args:
            output_path (Path): The path to the output directory.
        """
        self.setup("images")
        for batch, metadata in self.dataloader_images:
            for i, image in enumerate(batch):
                image = torch.swapaxes(image, 0, 2)
                image = Image.fromarray(
                    (numpy.clip(image.numpy(), 0, 1) * 255).astype(numpy.uint8),
                    mode="RGB",
                )
                output_path.mkdir(parents=True, exist_ok=True)
                filename = output_path / Path(metadata["subhalo_id"][i] + ".jpg")
                image.save(filename)

    def create_thumbnails(self, output_path: Path):
        """Writes preview images to disk.

        Args:
            output_path (Path): The path to the output directory.
        """
        self.setup("thumbnail_images")
        for batch, metadata in self.dataloader_thumbnail_images:
            for i, image in enumerate(batch):
                image = torch.swapaxes(image, 0, 2)
                image = Image.fromarray(
                    (numpy.clip(image.numpy(), 0, 1) * 255).astype(numpy.uint8),
                    mode="RGB",
                )
                output_path.mkdir(parents=True, exist_ok=True)
                filename = output_path / Path(metadata["subhalo_id"][i] + ".jpg")
                image.save(filename)

    def create_gas_pointclouds(self, output_path: Path):
        self.setup("gas_pointclouds")
        for batch, metadata in tqdm(self.dataloader_gas_pointclouds):
            for i, image in enumerate(batch):
                output_path.mkdir(parents=True, exist_ok=True)
                sid = metadata["subhalo_id"][i]
                filename = output_path / Path(f"{sid}.ply")
                o3d.io.write_point_cloud(str(filename), self.data_gas_pointclouds.get_visual_data(i))


