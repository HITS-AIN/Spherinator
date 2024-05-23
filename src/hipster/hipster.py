""" Provides all functionalities to transform a model in a HiPS representation for browsing.
"""

import copy
import math
import multiprocessing as mp
import os
from datetime import datetime
from pathlib import Path
from shutil import rmtree

import healpy
import numpy
import pandas as pd
import psutil
import torch
import torchvision.transforms.functional as functional
from astropy.io.votable import writeto
from astropy.table import Table
from PIL import Image

from spherinator.data import SpherinatorDataModule
from spherinator.models import SpherinatorModule

from .create_allsky import create_allsky


def create_embeded_tile(hipster, dataset, catalog, healpix_cells, i, range_j):
    for j in range_j:
        data = hipster.embed_tile(
            dataset, catalog, i, j, hipster.hierarchy, healpix_cells[j]
        )
        image = Image.fromarray(
            (numpy.clip(data.detach().numpy(), 0, 1) * 255).astype(numpy.uint8)
        )
        image.save(
            os.path.join(
                hipster.output_folder,
                hipster.title,
                "projection",
                "Norder" + str(i),
                "Dir" + str(int(math.floor(j / 10000)) * 10000),
                "Npix" + str(j) + ".jpg",
            )
        )
        print(".", end="", flush=True)


def create_hips_tile(hipster, model, i, range_j):
    for j in range_j:
        vectors = torch.zeros(
            (hipster.hierarchy**2, 3), dtype=torch.float32
        )  # prepare vor n*n subtiles
        for sub in range(
            hipster.hierarchy**2
        ):  # calculate coordinates for all n*n subpixels
            vector = healpy.pix2vec(
                2**i * hipster.hierarchy, j * hipster.hierarchy**2 + sub, nest=True
            )
            vectors[sub] = torch.tensor(vector).reshape(1, 3).type(dtype=torch.float32)
        with torch.no_grad():  # calculating for many subtile minimizes synchronisation overhead
            data = model.reconstruct(vectors)
        image = hipster.generate_tile(data, i, j, hipster.hierarchy, 0)
        image = Image.fromarray(
            (numpy.clip(image.detach().numpy(), 0, 1) * 255).astype(numpy.uint8)
        )
        image.save(
            os.path.join(
                hipster.output_folder,
                hipster.title,
                "model",
                "Norder" + str(i),
                "Dir" + str(int(math.floor(j / 10000)) * 10000),
                "Npix" + str(j) + ".jpg",
            )
        )
        print(".", end="", flush=True)


class Hipster:
    """
    Provides all functions to automatically generate a HiPS representation for a machine learning
    model that projects images on a sphere.
    """

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
        assert math.log2(output_size) == int(math.log2(output_size))
        assert max_order < 10
        self.output_folder = Path(output_folder)
        self.title = title
        self.title_folder = self.output_folder / Path(title)
        self.max_order = max_order
        self.hierarchy = hierarchy
        self.crop_size = crop_size
        self.output_size = output_size
        self.distortion_correction = distortion_correction
        self.catalog_file = self.title_folder / Path(catalog_file)
        self.votable_file = self.title_folder / Path(votable_file)
        self.hipster_url = hipster_url
        self.verbose = verbose

        if number_of_workers == -1:
            self.number_of_workers = psutil.cpu_count(logical=False)
        else:
            self.number_of_workers = number_of_workers

        if self.verbose > 0:
            print("number of workers: ", self.number_of_workers)

        self.title_folder.mkdir(parents=True, exist_ok=True)

    def check_folders(self, base_folder):
        """Checks whether the base folder exists and deletes it after prompting for user input

        Args:
            base_folder (String): The base folder to check.
        """
        path = os.path.join(self.output_folder, self.title, base_folder)
        if os.path.exists(path):
            answer = input("path " + str(path) + ", delete? Yes,[No]")
            if answer == "Yes":
                rmtree(os.path.join(self.output_folder, self.title, base_folder))
            else:
                exit(1)

    def create_folders(self, base_folder):
        """Creates all folders and sub-folders to store the HiPS tiles.

        Args:
            base_folder (String): The base folder to start the folder creation in.
        """
        print("creating folders:")
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)
        if not os.path.exists(os.path.join(self.output_folder, self.title)):
            os.mkdir(os.path.join(self.output_folder, self.title))
        os.mkdir(os.path.join(self.output_folder, self.title, base_folder))
        for i in range(self.max_order + 1):
            os.mkdir(
                os.path.join(
                    self.output_folder, self.title, base_folder, "Norder" + str(i)
                )
            )
            for j in range(int(math.floor(12 * 4**i / 10000)) + 1):
                os.mkdir(
                    os.path.join(
                        self.output_folder,
                        self.title,
                        base_folder,
                        "Norder" + str(i),
                        "Dir" + str(j * 10000),
                    )
                )

    def create_hips_properties(self, base_folder):
        """Generates the properties file that contains the meta-information of the HiPS tiling.

        Args:
            base_folder (String): The place where to create the 'properties' file.
        """
        print("creating meta-data:")
        with open(
            os.path.join(self.output_folder, self.title, base_folder, "properties"),
            "w",
            encoding="utf-8",
        ) as output:
            # TODO: add all keywords support and write proper information
            output.write("creator_did          = ivo://HITS/hipster\n")
            output.write("obs_title            = " + self.title + "\n")
            output.write("obs_description      = blablabla\n")
            output.write("dataproduct_type     = image\n")
            output.write("dataproduct_subtype  = color\n")
            output.write("hips_version         = 1.4\n")
            output.write("prov_progenitor      = blablabla\n")
            output.write("hips_creation_date   = " + datetime.now().isoformat() + "\n")
            output.write("hips_release_date    = " + datetime.now().isoformat() + "\n")
            output.write("hips_status          = public master clonable\n")
            output.write("hips_tile_format     = jpeg\n")
            output.write("hips_order           = " + str(self.max_order) + "\n")
            output.write(
                "hips_tile_width      = "
                + str(self.output_size * self.hierarchy)
                + "\n"
            )
            output.write("hips_frame           = equatorial\n")
            output.flush()

    def create_index_file(self, base_folder):
        """Generates the 'index.html' file that contains an direct access to the HiPS tiling via
            aladin lite.

        Args:
            base_folder (String): The place where to create the 'index.html' file.
        """
        print("creating index.html:")
        with open(
            os.path.join(self.output_folder, self.title, base_folder, "index.html"),
            "w",
            encoding="utf-8",
        ) as output:
            output.write("<!DOCTYPE html>\n")
            output.write("<html>\n")
            output.write("<head>\n")
            output.write(
                "<meta name='description' content='custom HiPS of "
                + self.title
                + "'>\n"
            )
            output.write("  <meta charset='utf-8'>\n")
            output.write(
                "  <title>HiPSter representation of " + self.title + "</title>\n"
            )
            output.write("</head>\n")
            output.write("<body>\n")
            output.write(
                "    <div id='aladin-lite-div' style='width:500px;height:500px;'></div>\n"
            )
            output.write(
                "    <script type='text/javascript' "
                + "src='https://aladin.u-strasbg.fr/AladinLite/api/v3/latest/aladin.js'"
                + "charset='utf-8'></script>\n"
            )
            output.write("    <script type='text/javascript'>\n")
            output.write("        var aladin;\n")
            output.write("	    A.init.then(() => {\n")
            output.write("            aladin = A.aladin('#aladin-lite-div');\n")
            # TODO: check this current hack for the tile location!!!
            output.write(
                "            aladin.setImageSurvey(aladin.createImageSurvey("
                + "'"
                + self.title
                + "', "
                + "'sphere projection of data from "
                + self.title
                + "', "
                + "'"
                + self.hipster_url
                + "/"
                + self.title
                + "/"
                + base_folder
                + "',"
                + "'equatorial', "
                + str(self.max_order)
                + ", {imgFormat: 'jpg'})); \n"
            )
            output.write("            aladin.setFoV(180.0); \n")
            output.write("        });\n")
            output.write("    </script>\n")
            output.write("</body>\n")
            output.write("</html>")
            output.flush()

    def calculate_pixels(self, matrix, pixel):
        size = matrix.shape[0]
        if size > 1:
            matrix[: size // 2, : size // 2] = self.calculate_pixels(
                matrix[: size // 2, : size // 2], pixel * 4
            )
            matrix[size // 2 :, : size // 2] = self.calculate_pixels(
                matrix[size // 2 :, : size // 2], pixel * 4 + 1
            )
            matrix[: size // 2, size // 2 :] = self.calculate_pixels(
                matrix[: size // 2, size // 2 :], pixel * 4 + 2
            )
            matrix[size // 2 :, size // 2 :] = self.calculate_pixels(
                matrix[size // 2 :, size // 2 :], pixel * 4 + 3
            )
        else:
            matrix = pixel
        return matrix

    def project_data(self, data, order, pixel):
        if not self.distortion_correction:
            data = functional.resize(
                data, [self.output_size, self.output_size], antialias=True
            )  # scale
            data = torch.swapaxes(data, 0, 2)
            return data
        data = torch.swapaxes(data, 0, 2)
        result = torch.zeros(
            (self.output_size, self.output_size, 3)
        )  # * torch.tensor((77.0/255.0, 0.0/255.0, 153.0/255.0)).reshape(3,1).T[:,None]
        healpix_pixel = torch.zeros(
            (self.output_size, self.output_size), dtype=torch.int64
        )
        healpix_pixel = self.calculate_pixels(healpix_pixel, pixel)
        center_theta, center_phi = healpy.pix2ang(
            2**order, pixel, nest=True
        )  # theta 0...180 phi 0...360
        size = data.shape[0]
        max_theta = max_phi = 2 * math.pi / (4 * 2**order) / 2
        for x in range(self.output_size):
            for y in range(self.output_size):
                target_theta, target_phi = healpy.pix2ang(
                    2**order * self.output_size, healpix_pixel[x, y], nest=True
                )
                delta_theta = target_theta - center_theta
                if center_phi == 0 and target_phi > math.pi:
                    delta_phi = (target_phi - center_phi - 2 * math.pi) * math.sin(
                        target_theta
                    )
                else:
                    delta_phi = (target_phi - center_phi) * math.sin(target_theta)
                target_x = int(size // 2 + delta_phi / max_phi * (size // 2 - 1))
                target_y = int(size // 2 + delta_theta / max_theta * (size // 2 - 1))
                if (
                    target_x >= 0
                    and target_y >= 0
                    and target_x < size
                    and target_y < size
                ):
                    result[x, y] = data[target_x, target_y]
                # else:
                #     result[x,y] = 0
        return result

    def generate_tile(self, data, order, pixel, hierarchy, index):
        if hierarchy <= 1:
            vector = healpy.pix2vec(2**order, pixel, nest=True)
            vector = torch.tensor(vector).reshape(1, 3).type(dtype=torch.float32)
            with torch.no_grad():
                reconstruction = data[index]  # model.reconstruct(vector)[0]
            return self.project_data(reconstruction, order, pixel)
        q1 = self.generate_tile(data, order + 1, pixel * 4, hierarchy / 2, index * 4)
        q2 = self.generate_tile(
            data, order + 1, pixel * 4 + 1, hierarchy / 2, index * 4 + 1
        )
        q3 = self.generate_tile(
            data, order + 1, pixel * 4 + 2, hierarchy / 2, index * 4 + 2
        )
        q4 = self.generate_tile(
            data, order + 1, pixel * 4 + 3, hierarchy / 2, index * 4 + 3
        )
        result = torch.ones((q1.shape[0] * 2, q1.shape[1] * 2, 3))
        result[: q1.shape[0], : q1.shape[1]] = q1
        result[q1.shape[0] :, : q1.shape[1]] = q2
        result[: q1.shape[0], q1.shape[1] :] = q3
        result[q1.shape[0] :, q1.shape[1] :] = q4
        return result

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
                + str(12 * 4**i).rjust(int(math.log10(12 * 4**self.max_order)) + 1, " ")
                + " tiles]:",
                end="",
                flush=True,
            )
            if self.number_of_workers == 1:
                create_hips_tile(
                    self, model, i, range(12 * 4**i // self.number_of_workers)
                )
            else:
                mypool = []
                for t in range(self.number_of_workers):
                    mypool.append(
                        mp.Process(
                            target=create_hips_tile,
                            args=(
                                self,
                                model,
                                i,
                                range(
                                    t * 12 * 4**i // self.number_of_workers,
                                    (t + 1) * 12 * 4**i // self.number_of_workers,
                                ),
                            ),
                        )
                    )
                    mypool[-1].start()
                for process in mypool:
                    process.join()
            print(" done", flush=True)
        print("done!")

    def transform_csv_to_votable(self):
        if self.verbose > 0:
            print("Transforming catalog.csv to votable ...")

        table = Table.read(self.catalog_file, format="ascii.csv")
        writeto(table, str(self.votable_file))

        if self.verbose > 0:
            print("Transforming catalog.csv to votable ... done.")

    def generate_catalog(
        self, model: SpherinatorModule, datamodule: SpherinatorDataModule
    ):
        """Generates a catalog by mapping all provided data using the encoder of the
            trained model. The catalog will contain the id, coordinates in angles as well
            as unity vector coordinates, the reconstruction loss, and a link to the original
            image file.

        Args:
            model (SpherinatorModule): A model that allows to call project_dataset(x) to encode elements
                of a dataset to a sphere.
            datamodule (SpherinatorDataModule): A datamodule to access the images that should get mapped.
        """

        if self.catalog_file.exists():
            answer = input("Catalog exists, overwrite? Yes,[No] ")
            if answer != "Yes":
                return

        print("Creating catalog csv-file ...")
        datamodule.write_catalog(model, self.catalog_file, self.hipster_url, self.title)

    def calculate_healpix_cells(self, catalog, numbers, order, pixels):
        healpix_cells = {}  # create an extra map to quickly find images in a cell
        for pixel in pixels:
            healpix_cells[pixel] = []  # create empty lists for each cell
        for number in numbers:
            pixel = healpy.vec2pix(
                2**order,
                catalog[number][2],
                catalog[number][3],
                catalog[number][4],
                nest=True,
            )
            if pixel in healpix_cells:
                healpix_cells[pixel].append(int(number))
        return healpix_cells

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
                    numpy.square(catalog[numpy.array(idx)][:, 4:7] - vector), axis=1
                )
                best = idx[numpy.argmin(distances)]
                data, _ = dataset[int(catalog[best][0])]
                data = functional.rotate(data, catalog[best][1], expand=False)
                data = functional.center_crop(
                    data, [self.crop_size, self.crop_size]
                )  # crop
                data = self.project_data(data, order, pixel)
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
            usecols=["id", "rotation", "x", "y", "z"],
        ).to_numpy()

        for i in range(self.max_order + 1):
            healpix_cells = self.calculate_healpix_cells(
                catalog, range(catalog.shape[0]), i, range(12 * 4**i)
            )
            print(
                "\n  order "
                + str(i)
                + " ["
                + str(12 * 4**i).rjust(int(math.log10(12 * 4**self.max_order)) + 1, " ")
                + " tiles]:",
                end="",
            )

            mypool = []

            # for j in range(12*4**i):
            #     if j % (int(12*4**i/100)+1) == 0:
            #         print(".", end="", flush=True)
            #     data = self.embed_tile(dataset, catalog, i, j, self.hierarchy, healpix_cells[j])
            #     image = Image.fromarray((
            #         numpy.clip(data.detach().numpy(),0,1)*255).astype(numpy.uint8))
            #     image.save(os.path.join(self.output_folder,
            #                             self.title,
            #                             "projection",
            #                             "Norder"+str(i),
            #                             "Dir"+str(int(math.floor(j/10000))*10000),
            #                             "Npix"+str(j)+".jpg"))

            for t in range(self.number_of_workers):
                mypool.append(
                    mp.Process(
                        target=create_embeded_tile,
                        args=(
                            self,
                            copy.deepcopy(dataset),
                            copy.deepcopy(catalog),
                            copy.deepcopy(healpix_cells),
                            i,
                            range(
                                t * 12 * 4**i // self.number_of_workers,
                                (t + 1) * 12 * 4**i // self.number_of_workers,
                            ),
                        ),
                    )
                )
                mypool[-1].start()
            for process in mypool:
                process.join()
            print(" done", flush=True)

        if self.verbose > 0:
            print("Generating dataset projection ... done.")

    def create_images(self, datamodule: SpherinatorDataModule):
        output_path = self.title_folder / Path("jpg")
        output_path.mkdir(parents=True, exist_ok=True)
        datamodule.create_images(output_path)

    def create_thumbnails(self, datamodule: SpherinatorDataModule):
        output_path = self.title_folder / Path("thumbnails")
        output_path.mkdir(parents=True, exist_ok=True)
        datamodule.create_thumbnails(output_path)

    def create_allsky(self):
        if self.verbose > 0:
            print("Create allsky images ...")
        create_allsky(self.title_folder / "model", max_order=self.max_order)
        create_allsky(self.title_folder / "projection", max_order=self.max_order)
        if self.verbose > 0:
            print("Create allsky images ... done.")
