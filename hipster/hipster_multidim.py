from pathlib import Path

import healpy
import torch
from PIL import Image
import math
import numpy
import pandas as pd
import torchvision.transforms.functional as functional

from data.spherinator_data_module import SpherinatorDataModule
from models.spherinator_module import SpherinatorModule
from .hipster import Hipster
import os


class HipsterMultidim(Hipster):

    def __init__(
        self,
        output_folder: str,
        title: str,
        max_order: int = 3,
        hierarchy: int = 1,
        crop_size: int = 64,
        output_size: int = 128,
        distortion_correction: bool = False,
        number_of_workers: int = -1,
        catalog_file: str = "catalog.csv",
        votable_file: str = "catalog.vot",
        hipster_url: str = "http://localhost:8082",
        verbose: int = 0,
    ):
        """Initializes the Hipster

        Args:
            output_folder (String): The place where to export the HiPS to. In case it exists, there
                is a user prompt before deleting the folder.
            title (String): The title string to be passed to the meta files.
            max_order (int, optional): The depth of the tiling. Should be smaller than 10.
                Defaults to 3.
            hierarchy: (int, optional): Defines how many tiles should be hierarchically combined.
                Defaults to 1.
            crop_size (int, optional): The size to be cropped from the generating model output, in
                case it might be larger. Defaults to 64.
            output_size (int, optional): Specifies the size the tilings should be scaled to. Must be
                in the powers of 2. Defaults to 128.
            distortion_correction (bool, optional): Wether or not to apply a distortion correction
            number_of_workers (int, optional): The number of CPU threads. Defaults to -1, which means
                all available threads.
            catalog_file (String, optional): The name of the catalog file to be generated.
                Defaults to "catalog.csv".
            votable_file (String, optional): The name of the votable file to be generated.
                Defaults to "catalog.vot".
            hipster_url (String, optional): The url where the HiPSter will be hosted.
                Defaults to "http://localhost:8082".
            verbose (int, optional): The verbosity level. Defaults to 0.
        """
        super().__init__(output_folder, title, max_order, hierarchy, crop_size, output_size, distortion_correction,
                         number_of_workers, catalog_file, votable_file, hipster_url, verbose)
        self.healpix_mapping = {}

    def generate_hips(self, model: SpherinatorModule):
        """Generates a HiPS tiling following the standard defined in
            https://www.ivoa.net/documents/HiPS/20170519/REC-HIPS-1.0-20170519.pdf

        Args:
            model (SpherinatorModule): A model that allows to call decode(x) for a three dimensional
            vector x. The resulting reconstructions are used to generate the tiles for HiPS.
        """
        self.check_folders("model")
        self.create_folders("model")
        self.create_hips_properties("model")
        self.create_index_file("model")
        print("creating tiles:")
        for i in range(self.max_order + 1):
            print(
                "  order "
                + str(i)
                + " ["
                + str(12 * 4**i).rjust(
                    int(math.log10(12 * 4**self.max_order)) + 1, " "
                )
                + " tiles]:",
                end="",
                flush=True,
            )
            for j in range(12 * 4 ** i):
                if j % (int(12 * 4 ** i / 100) + 1) == 0:
                    print(".", end="", flush=True)
                vectors = torch.zeros((self.hierarchy**2, 3), dtype=torch.float32)  # prepare vor n*n subtiles
                for sub in range(self.hierarchy**2):  # calculate coordinates for all n*n subpixels
                    vector = healpy.pix2vec(2**i * self.hierarchy, j * self.hierarchy**2 + sub, nest=True)
                    vectors[sub] = torch.tensor(vector).reshape(1, 3).type(dtype=torch.float32)
                with torch.no_grad():  # calculating for many subtile minimizes synchronisation overhead
                    data = model.reconstruct(vectors)
                image = self.generate_tile(data, i, j, self.hierarchy, 0)
                image = Image.fromarray((numpy.clip(image.detach().numpy(), 0, 1) * 255).astype(numpy.uint8))
                image.save(os.path.join(self.output_folder,
                                        self.title,
                                        "model",
                                        "Norder" + str(i),
                                        "Dir" + str(int(math.floor(j / 10000)) * 10000),
                                        "Npix" + str(j) + ".jpg",
                                        ))
                print(".", end="", flush=True)
        print("done!")

    def embed_tile(self, dataset, catalog, order, pixel, hierarchy, idx):
        if hierarchy <= 1:
            if len(idx) == 0:
                data = torch.ones((3, self.output_size, self.output_size))
                data[0] = data[0] * 77.0 / 255.0  # deep purple
                data[1] = data[1] * 0.0 / 255.0
                data[2] = data[2] * 153.0 / 255.0
                data = torch.swapaxes(data, 0, 2)
            else:
                vector = healpy.pix2vec(2**order, pixel, nest=True)
                distances = numpy.sum(
                    numpy.square(catalog[numpy.array(idx)][:, 5:8] - vector), axis=1
                )
                best = idx[numpy.argmin(distances)]
                data, _ = dataset[int(catalog[best][1])]
                data = functional.rotate(data, catalog[best][2], expand=False)
                data = functional.center_crop(
                    data, [self.crop_size, self.crop_size]
                )  # crop
                data = self.project_data(data, order, pixel)
                subhalo_id = catalog[best][0]
            return data
        healpix_cells = self.calculate_healpix_cells(
            catalog, idx, order + 1, range(pixel * 4, pixel * 4 + 4)
        )
        q1 = self.embed_tile(
            dataset,
            catalog,
            order + 1,
            pixel * 4,
            hierarchy / 2,
            healpix_cells[pixel * 4],
        )
        q2 = self.embed_tile(
            dataset,
            catalog,
            order + 1,
            pixel * 4 + 1,
            hierarchy / 2,
            healpix_cells[pixel * 4 + 1],
        )
        q3 = self.embed_tile(
            dataset,
            catalog,
            order + 1,
            pixel * 4 + 2,
            hierarchy / 2,
            healpix_cells[pixel * 4 + 2],
        )
        q4 = self.embed_tile(
            dataset,
            catalog,
            order + 1,
            pixel * 4 + 3,
            hierarchy / 2,
            healpix_cells[pixel * 4 + 3],
        )
        result = torch.ones((q1.shape[0] * 2, q1.shape[1] * 2, 3))
        result[: q1.shape[0], : q1.shape[1]] = q1
        result[q1.shape[0] :, : q1.shape[1]] = q2
        result[: q1.shape[0], q1.shape[1] :] = q3
        result[q1.shape[0] :, q1.shape[1] :] = q4
        return result

    def generate_dataset_projection(self, datamodule: SpherinatorDataModule):
        """Generates a HiPS tiling by using the coordinates of every image to map the original
            images form the data set based on their distance to the closest heal pixel cell
            center.

        Args:
            datamodule (SpherinatorDataModule): The datamodule to access the original images
        """
        if self.verbose > 0:
            print("Generating dataset projection ...")

        self.check_folders("projection")
        self.create_folders("projection")
        self.create_hips_properties("projection")
        self.create_index_file("projection")

        datamodule.setup("processing")
        dataset = datamodule.data_processing

        catalog = pd.read_csv(
            self.catalog_file,
            usecols=[ "subhalo_id", "id", "rotation", "x", "y", "z"],
        ).to_numpy()

        for i in range(self.max_order + 1):
            healpix_cells = self.calculate_healpix_cells(
                catalog, range(catalog.shape[0]), i, range(12 * 4**i)
            )
            print(
                "\n  order "
                + str(i)
                + " ["
                + str(12 * 4**i).rjust(
                    int(math.log10(12 * 4**self.max_order)) + 1, " "
                )
                + " tiles]:",
                end="",
            )


            for j in range(12*4**i):
                if j % (int(12*4**i/100)+1) == 0:
                    print(".", end="", flush=True)
                data = self.embed_tile(dataset, catalog, i, j, self.hierarchy, healpix_cells[j])
                image = Image.fromarray((numpy.clip(data.detach().numpy(),0,1)*255).astype(numpy.uint8))
                image.save(os.path.join(self.output_folder,
                                        self.title,
                                        "projection",
                                        "Norder"+str(i),
                                        "Dir"+str(int(math.floor(j/10000))*10000),
                                        "Npix"+str(j)+".jpg"))



        if self.verbose > 0:
            print("Generating dataset projection ... done.")

    def add_mapping(self, subhalo_id, order, pixel,  ra, dec):
        self.healpix_mapping[str(subhalo_id)] = {
            "Norder": order,
            "Npix": pixel,
            "R"
        }

    def create_images(self, datamodule: SpherinatorDataModule, output_path=None):
        if output_path is None:
            output_path = self.title_folder
        output_path = output_path / Path("jpg")
        output_path.mkdir(parents=True, exist_ok=True)
        datamodule.create_images(output_path)

    def create_gas_pointclouds(self, datamodule, output_path=None):
        if output_path is None:
            output_path = self.title_folder
        output_path = output_path / Path("gasclouds")
        output_path.mkdir(parents=True, exist_ok=True)
        datamodule.create_gas_pointclouds(output_path)

    def create_data_cube(self, datamodule, data_aspects):
        output_path = self.title_folder / Path("data_cube")
        for aspect in data_aspects:
            create_visuals = getattr(self, f"create_{aspect}")
            create_visuals(datamodule, output_path)
