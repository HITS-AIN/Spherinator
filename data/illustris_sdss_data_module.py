import math
from pathlib import Path

import healpy
import numpy
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import data.preprocessing as preprocessing
from data.illustris_sdss_dataset import IllustrisSdssDataset
from data.illustris_sdss_dataset_with_metadata import IllustrisSdssDatasetWithMetadata
from models.spherinator_module import SpherinatorModule

from .spherinator_data_module import SpherinatorDataModule


class IllustrisSdssDataModule(SpherinatorDataModule):
    """Defines access to the Illustris sdss data as a data module."""

    def __init__(
        self,
        data_directories: list[str],
        extension: str = "fits",
        minsize: int = 100,
        shuffle: bool = True,
        batch_size: int = 32,
        num_workers: int = 16,
    ):
        """Initialize IllustrisSdssDataModule.

        Args:
            data_directories (list[str]): The directories to scan for data files.
            extension (str, optional): The kind of files to search for. Defaults to "fits".
            minsize (int, optional): The minimum size a file should have. Defaults to 100 pixels.
            shuffle (bool, optional): Wether or not to shuffle whe reading. Defaults to True.
            batch_size (int, optional): The batch size for training. Defaults to 32.
            num_workers (int, optional): How many worker to use for loading. Defaults to 16.
        """
        super().__init__()

        self.data_directories = data_directories
        self.extension = extension
        self.minsize = minsize
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.project_url = "https://www.tng-project.org"

        self.transform_images = transforms.Compose(
            [
                preprocessing.CreateNormalizedRGBColors(
                    stretch=0.9,
                    range=5,
                    lower_limit=0.001,
                    channel_combinations=[[2, 3], [1, 0], [0]],
                    scalers=[0.7, 0.5, 1.3],
                ),
            ]
        )
        self.transform_processing = transforms.Compose(
            [
                transforms.CenterCrop((363, 363)),
                self.transform_images,
            ]
        )
        self.transform_train = transforms.Compose(
            [
                self.transform_processing,
                preprocessing.DielemanTransformation(
                    rotation_range=[0, 360],
                    translation_range=[0, 0],  # 4./363,4./363],
                    scaling_range=[1, 1],  # 0.9,1.1],
                    flip=0.5,
                ),
                transforms.CenterCrop((363, 363)),
            ]
        )
        self.transform_thumbnail_images = transforms.Compose(
            [
                transforms.CenterCrop((363, 363)),
                transforms.Resize((100, 100), antialias=True),
                self.transform_images,
            ]
        )

    def setup(self, stage: str):
        """Sets up the data set and data loaders.

        Args:
            stage (str): Defines for which stage the data is needed.
        """
        if not stage in ["fit", "processing", "images", "thumbnail_images"]:
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
            self.data_processing = IllustrisSdssDatasetWithMetadata(
                data_directories=self.data_directories,
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
            self.data_images = IllustrisSdssDataset(
                data_directories=self.data_directories,
                extension=self.extension,
                minsize=self.minsize,
                transform=self.transform_images,
            )
            self.dataloader_images = DataLoader(
                self.data_images,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
            )
        elif stage == "thumbnail_images" and self.data_thumbnail_images is None:
            self.data_thumbnail_images = IllustrisSdssDataset(
                data_directories=self.data_directories,
                extension=self.extension,
                minsize=self.minsize,
                transform=self.transform_thumbnail_images,
            )
            self.dataloader_thumbnail_images = DataLoader(
                self.data_thumbnail_images,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
            )

    def write_catalog(
        self, model: SpherinatorModule, catalog_file: Path, hipster_url: str, title: str
    ):
        """Writes a catalog csv-file to disk.

        Args:
            model (SpherinatorModule): The model to use for the catalog.
            catalog_file (Path): The path to the catalog file.
            hipster_url (str): The domain to use for the HiPSter.
            title (str): The title to use for the catalog.
        """
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
                    output.write(str(i) + ",")
                    output.write(str(angles[i, 1]) + ",")
                    output.write(str(90.0 - angles[i, 0]) + ",")
                    output.write(str(rotations[i]) + ",")
                    output.write(str(coordinates[i, 0]) + ",")
                    output.write(str(coordinates[i, 1]) + ",")
                    output.write(str(coordinates[i, 2]) + "\n")

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
                filename = output_path / Path(
                    metadata["simulation"][i],
                    metadata["snapshot"][i],
                    metadata["subhalo_id"][i] + ".jpg",
                )
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
                filename = output_path / Path(
                    metadata["simulation"][i],
                    metadata["snapshot"][i],
                    metadata["subhalo_id"][i] + ".jpg",
                )
                image.save(filename)
