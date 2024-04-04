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
from matplotlib import pyplot as plt

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
        elif stage == "gas_pointclouds":
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
        elif stage == "dm_pointclouds":
            self.data_dm_pointclouds = IllustrisSdssDatasetMultidim(
                data_directories=self.data_directories,
                cutout_directory=self.cutout_directory,
                info_dir=self.info_directory,
                data_aspect="dm_pointcloud",
                extension=self.extension,
                minsize=self.minsize,
                transform=self.transform_processing,
            )
            self.dataloader_dm_pointclouds = DataLoader(
                self.data_dm_pointclouds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        elif stage == "stars":
            self.data_stars = IllustrisSdssDatasetMultidim(
                data_directories=self.data_directories,
                cutout_directory=self.cutout_directory,
                info_dir=self.info_directory,
                data_aspect="stars",
                extension=self.extension,
                minsize=self.minsize,
                transform=self.transform_processing,
            )
            self.dataloader_stars = DataLoader(
                self.data_stars,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        elif stage == "gas_temperature_fields":
            self.data_gas_temperature_fields = IllustrisSdssDatasetMultidim(
                data_directories=self.data_directories,
                cutout_directory=self.cutout_directory,
                info_dir=self.info_directory,
                data_aspect="gas_temperature_field",
                extension=self.extension,
                minsize=self.minsize,
                transform=self.transform_processing,
            )
            self.dataloader_gas_temperature_fields = DataLoader(
                self.data_gas_temperature_fields,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        elif stage == "dark_matter_fields":
            self.data_dark_matter_fields = IllustrisSdssDatasetMultidim(
                data_directories=self.data_directories,
                cutout_directory=self.cutout_directory,
                info_dir=self.info_directory,
                data_aspect="dark_matter_field",
                extension=self.extension,
                minsize=self.minsize,
                transform=self.transform_processing,
            )
            self.dataloader_dark_matter_fields = DataLoader(
                self.data_dark_matter_fields,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

    def write_catalog(
        self, model: SpherinatorModule, catalog_file: Path, hipster_url: str, title: str
    ):
        # Todo: Needs adjustment
        self.setup("processing")
        with open(catalog_file, "w", encoding="utf-8") as output:
            output.write(
                "#preview,simulation,snapshot data,subhalo id,subhalo data,RMSE,id,RA2000,DEC2000,rotation,x,y,z\n"
            )

            for batch, metadata in tqdm(self.dataloader_processing):
                _, rotations, coordinates, losses = model.find_best_rotation(batch)

                rotations = rotations.cpu().detach().numpy()
                coordinates = coordinates.cpu().detach().numpy()
                losses = losses.cpu().detach().numpy()
                angles = numpy.array(healpy.vec2ang(coordinates)) * 180.0 / math.pi
                angles = angles.T

                for i in range(len(batch)):
                    output.write("<a href='" + hipster_url + "/" + title + "/jpg/")
                    output.write(str(metadata["simulation"][i]) + "/")
                    output.write(str(metadata["snapshot"][i]) + "/")
                    output.write(
                        str(metadata["subhalo_id"][i]) + ".jpg' target='_blank'>"
                    )
                    output.write(
                        "<img src='" + hipster_url + "/" + title + "/thumbnails/"
                    )
                    output.write(str(metadata["simulation"][i]) + "/")
                    output.write(str(metadata["snapshot"][i]) + "/")
                    output.write(str(metadata["subhalo_id"][i]) + ".jpg'></a>,")
                    output.write(str(metadata["simulation"][i]) + ",")
                    output.write(str(metadata["snapshot"][i]) + ",")
                    output.write(str(metadata["subhalo_id"][i]) + ",")
                    output.write("<a href='" + self.project_url + "/api/")
                    output.write(str(metadata["simulation"][i]) + "-1/snapshots/")
                    output.write(str(metadata["snapshot"][i]) + "/subhalos/")
                    output.write(str(metadata["subhalo_id"][i]) + "/")
                    output.write("' target='_blank'>" + self.project_url + "</a>,")
                    output.write(str(losses[i]) + ",")
                    output.write(str(metadata["id"][i].item()) + ",")
                    output.write(str(angles[i, 1]) + ",")
                    output.write(str(90.0 - angles[i, 0]) + ",")
                    output.write(str(rotations[i]) + ",")
                    output.write(str(coordinates[i, 0]) + ",")
                    output.write(str(coordinates[i, 1]) + ",")
                    output.write(str(coordinates[i, 2]) + "\n")

    def create_morphology(self, output_path: Path):
        # Just a wrapper for my own naming convention
        self.create_images(output_path)

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
                filename = "{simulation}_{snapshot}_{subhalo_id}.jpg".format(
                    simulation=metadata["simulation"][i],
                    snapshot=metadata["snapshot"][i],
                    subhalo_id=metadata["subhalo_id"][i])
                image.save(output_path / Path(filename))

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
                filename = "{simulation}_{snapshot}_{subhalo_id}.jpg".format(
                    simulation=metadata["simulation"][i],
                    snapshot=metadata["snapshot"][i],
                    subhalo_id=metadata["subhalo_id"][i])
                image.save(output_path / Path(filename))

    def create_gas_pointclouds(self, output_path: Path):
        self.setup("gas_pointclouds")
        for batch, metadata in tqdm(self.dataloader_gas_pointclouds):
            for i, image in enumerate(batch):
                output_path.mkdir(parents=True, exist_ok=True)
                filename = "{simulation}_{snapshot}_{subhalo_id}.ply".format(
                    simulation=metadata["simulation"][i],
                    snapshot=metadata["snapshot"][i],
                    subhalo_id=metadata["subhalo_id"][i])
                output_name = str(output_path / Path(filename))
                o3d.io.write_point_cloud(output_name, self.data_gas_pointclouds.get_visual_data(i))

    def create_dm_pointclouds(self, output_path: Path):
        self.setup("dm_pointclouds")
        for batch, metadata in tqdm(self.dataloader_dm_pointclouds):
            for i, image in enumerate(batch):
                output_path.mkdir(parents=True, exist_ok=True)
                filename = "{simulation}_{snapshot}_{subhalo_id}.ply".format(
                    simulation=metadata["simulation"][i],
                    snapshot=metadata["snapshot"][i],
                    subhalo_id=metadata["subhalo_id"][i])
                output_name = str(output_path / Path(filename))
                o3d.io.write_point_cloud(output_name, self.data_dm_pointclouds.get_visual_data(i))

    def create_gas_temperature_fields(self, output_path: Path):
        self.setup("gas_temperature_fields")
        for batch, metadata in tqdm(self.dataloader_gas_temperature_fields):
            for i, image in enumerate(batch):
                output_path.mkdir(parents=True, exist_ok=True)
                filename = "{simulation}_{snapshot}_{subhalo_id}.png".format(
                    simulation=metadata["simulation"][i],
                    snapshot=metadata["snapshot"][i],
                    subhalo_id=metadata["subhalo_id"][i])
                vis_data, extent = self.data_gas_temperature_fields.get_visual_data(i)
                plt.figure()
                plt.imshow(vis_data, extent=extent)
                plt.xlabel("Distance from Galactic center [kpc]")
                plt.ylabel("Distance from Galactic center [kpc]")
                plt.colorbar(label="Gas Temperature [log(K)]")
                plt.savefig(str(output_path / filename))

    def create_dark_matter_fields(self, output_path: Path):
        self.setup("dark_matter_fields")
        for batch, metadata in tqdm(self.dataloader_dark_matter_fields):
            for i, image in enumerate(batch):
                output_path.mkdir(parents=True, exist_ok=True)
                filename = "{simulation}_{snapshot}_{subhalo_id}.png".format(
                    simulation=metadata["simulation"][i],
                    snapshot=metadata["snapshot"][i],
                    subhalo_id=metadata["subhalo_id"][i])
                vis_data, extent = self.data_dark_matter_fields.get_visual_data(i)
                plt.figure()
                plt.imshow(vis_data, extent=extent)
                plt.xlabel("Distance from Galactic center [kpc]")
                plt.ylabel("Distance from Galactic center [kpc]")
                plt.colorbar(label="Dark Matter Density [log(Msun / kpc)]")
                plt.savefig(str(output_path / filename))

    def create_stars(self, output_path: Path):
        self.setup("stars")
        for batch, metadata in tqdm(self.dataloader_stars):
            for i, image in enumerate(batch):
                output_path.mkdir(parents=True, exist_ok=True)
                filename = "{simulation}_{snapshot}_{subhalo_id}.ply".format(
                    simulation=metadata["simulation"][i],
                    snapshot=metadata["snapshot"][i],
                    subhalo_id=metadata["subhalo_id"][i])
                output_name = str(output_path / Path(filename))
                o3d.io.write_point_cloud(output_name, self.data_stars.get_visual_data(i))







