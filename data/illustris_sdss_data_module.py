from pathlib import Path

import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader

import data.preprocessing as preprocessing
from data.illustris_sdss_dataset import IllustrisSdssDataset
from models.spherinator_module import SpherinatorModule
from tqdm import tqdm

from .spherinator_data_module import SpherinatorDataModule


class IllustrisSdssDataModule(SpherinatorDataModule):
    """Defines access to the Illustris sdss data as a data module."""

    transform_images = transforms.Compose(
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
    transform_processing = transforms.Compose(
        [
            transforms.CenterCrop((363, 363)),
            transform_images,
        ]
    )
    transform_train = transforms.Compose(
        [
            transform_processing,
            preprocessing.DielemanTransformation(
                rotation_range=[0, 360],
                translation_range=[0, 0],  # 4./363,4./363],
                scaling_range=[1, 1],  # 0.9,1.1],
                flip=0.5,
            ),
            transforms.CenterCrop((363, 363)),
        ]
    )
    transform_thumbnail_images = transforms.Compose(
        [
            transforms.CenterCrop((363, 363)),
            transforms.Resize((100, 100), antialias=True),
            transform_images,
        ]
    )

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
            self.data_processing = IllustrisSdssDataset(
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

    def write_catalog(self, model: SpherinatorModule, catalog_file: Path):
        """Writes a catalog to disk."""
        self.setup("processing")
        with open(catalog_file, "w", encoding="utf-8") as output:
            output.write(
                "#preview,simulation,snapshot data,subhalo id,subhalo data,RMSE,id,RA2000,DEC2000,rotation,x,y,z\n"
            )

            for batch in tqdm(self.dataloader_processing):
                _, rotations, coordinates, losses = model.find_best_rotation(batch)

                output.write("<a href='https://space.h-its.org/Illustris/jpg/")
                output.write(str(dataloader.dataset[i]["metadata"]["simulation"]) + "/")
                output.write(str(dataloader.dataset[i]["metadata"]["snapshot"]) + "/")
                output.write(
                    str(dataloader.dataset[i]["metadata"]["subhalo_id"])
                    + ".jpg' target='_blank'>"
                )
                output.write("<img src='https://space.h-its.org/Illustris/thumbnails/")
                output.write(str(dataloader.dataset[i]["metadata"]["simulation"]) + "/")
                output.write(str(dataloader.dataset[i]["metadata"]["snapshot"]) + "/")
                output.write(
                    str(dataloader.dataset[i]["metadata"]["subhalo_id"]) + ".jpg'></a>,"
                )

                output.write(str(dataloader.dataset[i]["metadata"]["simulation"]) + ",")
                output.write(str(dataloader.dataset[i]["metadata"]["snapshot"]) + ",")
                output.write(str(dataloader.dataset[i]["metadata"]["subhalo_id"]) + ",")
                output.write("<a href='")
                output.write("https://www.tng-project.org/api/")
                output.write(
                    str(dataloader.dataset[i]["metadata"]["simulation"])
                    + "-1/snapshots/"
                )
                output.write(
                    str(dataloader.dataset[i]["metadata"]["snapshot"]) + "/subhalos/"
                )
                output.write(str(dataloader.dataset[i]["metadata"]["subhalo_id"]) + "/")
                output.write("' target='_blank'>www.tng-project.org</a>,")
                output.write(str(losses[i]) + ",")
                output.write(
                    str(i)
                    + ","
                    + str(angles[i, 1])
                    + ","
                    + str(90.0 - angles[i, 0])
                    + ","
                    + str(rotations[i])
                    + ","
                )
                output.write(
                    str(coordinates[i, 0])
                    + ","
                    + str(coordinates[i, 1])
                    + ","
                    + str(coordinates[i, 2])
                    + "\n"
                )
