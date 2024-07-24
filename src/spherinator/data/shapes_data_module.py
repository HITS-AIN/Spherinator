import math
from pathlib import Path
from typing import Union

import healpy
import numpy
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from spherinator.models.spherinator_module import SpherinatorModule

from .shapes_dataset import ShapesDataset
from .shapes_dataset_with_metadata import ShapesDatasetWithMetadata
from .spherinator_data_module import SpherinatorDataModule


class ShapesDataModule(SpherinatorDataModule):
    """Defines access to the ShapesDataset."""

    def __init__(
        self,
        data_directory: str,
        exclude_files: Union[list[str], str] = [],
        shuffle: bool = True,
        image_size: int = 91,
        batch_size: int = 32,
        num_workers: int = 1,
        download: bool = False,
    ):
        """Initializes the data loader

        Args:
            data_directory (str): The data directory
            exclude_files (list[str] | str, optional): A list of files to exclude. Defaults to [].
            shuffle (bool, optional): Wether or not to shuffle whe reading. Defaults to True.
            image_size (int, optional): The size of the images. Defaults to 91.
            batch_size (int, optional): The batch size for training. Defaults to 32.
            num_workers (int, optional): How many worker to use for loading. Defaults to 1.
            download (bool, optional): Wether or not to download the data. Defaults to False.
        """
        super().__init__()

        self.data_directory = data_directory
        self.exclude_files = exclude_files
        self.shuffle = shuffle
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download

        self.transform_train = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Resize((self.image_size, self.image_size), antialias=True),
                transforms.Lambda(  # Normalize
                    lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x))
                ),
            ]
        )
        self.transform_processing = self.transform_train
        self.transform_images = self.transform_train
        self.transform_thumbnail_images = transforms.Compose(
            [
                self.transform_train,
                transforms.Resize((100, 100), antialias=True),
            ]
        )

    def setup(self, stage: str):
        """Sets up the data set and data loaders.

        Args:
            stage (str): Defines for which stage the data is needed.
                         For the moment just fitting is supported.
        """
        if stage not in ["fit", "processing", "images", "thumbnail_images"]:
            raise ValueError(f"Stage {stage} not supported.")

        if stage == "fit" and self.data_train is None:
            self.data_train = ShapesDataset(
                data_directory=self.data_directory,
                exclude_files=self.exclude_files,
                transform=self.transform_train,
                download=self.download,
            )
            self.dataloader_train = DataLoader(
                self.data_train,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
            )
        elif stage == "processing" and self.data_processing is None:
            self.data_processing = ShapesDatasetWithMetadata(
                data_directory=self.data_directory,
                exclude_files=self.exclude_files,
                transform=self.transform_processing,
            )
            self.dataloader_processing = DataLoader(
                self.data_processing,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        elif stage == "images" and self.data_images is None:
            self.data_images = ShapesDatasetWithMetadata(
                data_directory=self.data_directory,
                exclude_files=self.exclude_files,
                transform=self.transform_images,
            )
            self.dataloader_images = DataLoader(
                self.data_images,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
            )
        elif stage == "thumbnail_images" and self.data_thumbnail_images is None:
            self.data_thumbnail_images = ShapesDatasetWithMetadata(
                data_directory=self.data_directory,
                exclude_files=self.exclude_files,
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
        """Writes a catalog to disk.

        Args:
            model (SpherinatorModule): The model to use for the catalog.
            catalog_file (Path): The path to the catalog file.
            hipster_url (str): The domain to use for the HiPSter.
            title (str): The title to use for the catalog.
        """
        self.setup("processing")
        with open(catalog_file, "w", encoding="utf-8") as output:
            output.write("#preview,RMSE,id,RA2000,DEC2000,rotation,x,y,z\n")

            for batch, metadata in tqdm(self.dataloader_processing):
                _, rotations, coordinates, losses = model.find_best_rotation(batch)

                rotations = rotations.cpu().detach().numpy()
                coordinates = coordinates.cpu().detach().numpy()
                losses = losses.cpu().detach().numpy()
                angles = numpy.array(healpy.vec2ang(coordinates)) * 180.0 / math.pi
                angles = angles.T

                for i in range(len(batch)):
                    output.write("<a href='" + hipster_url + "/" + title + "/jpg/")
                    output.write(
                        str(metadata["filename"][i]) + ".jpg' target='_blank'>"
                    )
                    output.write(
                        "<img src='" + hipster_url + "/" + title + "/thumbnails/"
                    )
                    output.write(str(metadata["filename"][i]) + ".jpg'></a>,")
                    output.write(str(losses[i]) + ",")
                    output.write(str(metadata["id"][i]) + ",")
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
                    metadata["shape"][i] + "_" + metadata["id"][i] + ".jpg",
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
                    metadata["shape"][i] + "_" + metadata["id"][i] + ".jpg",
                )
                image.save(filename)
