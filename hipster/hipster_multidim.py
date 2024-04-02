from datetime import datetime
from pathlib import Path

import healpy
import torch
from PIL import Image
import math
import numpy
import pandas as pd
import torchvision.transforms.functional as functional
from strenum import StrEnum

from data.spherinator_data_module import SpherinatorDataModule
from models.spherinator_module import SpherinatorModule
from .hipster import Hipster
import os

class SurveyType(StrEnum):
    MODEL = "model"
    PROJECTION = "projection"
    CAT = "interaction_catalog"


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

    def create_hips_properties(self, base_folder):
        """Generates the properties file that contains the meta-information of the HiPS tiling.

        Args:
            base_folder (String): The place where to create the 'properties' file.
        """
        print("creating meta-data:")
        is_cat = base_folder == SurveyType.CAT
        if is_cat:
            tile_format = 'tsv'
            dataproduct = 'catalog'
        else:
            tile_format = 'jpeg'
            dataproduct = 'image'

        with open(
            os.path.join(self.output_folder, self.title, base_folder, "properties"),
            "w",
            encoding="utf-8",
        ) as output:
            # TODO: add all keywords support and write proper information
            output.write("creator_did          = ivo://HITS/hipster\n")
            output.write("obs_title            = " + self.title + "\n")
            output.write("obs_description      = blablabla\n")
            output.write("dataproduct_type     = {0}\n".format(dataproduct))
            if not is_cat:
                output.write("dataproduct_subtype  = color\n")
            output.write("hips_version         = 1.4\n")
            output.write("prov_progenitor      = blablabla\n")
            output.write("hips_creation_date   = " + datetime.now().isoformat() + "\n")
            output.write("hips_release_date    = " + datetime.now().isoformat() + "\n")
            output.write("hips_status          = public master clonable\n")
            output.write("hips_tile_format     = {0}\n".format(tile_format))
            output.write("hips_order           = " + str(self.max_order) + "\n")
            output.write("hips_tile_width      = 512\n")
            output.write("hips_frame           = equatorial\n")
            output.flush()

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

    def embed_tile(self, dataset, catalog, order, pixel, hierarchy, idx, hier_catalog=None):
        if hier_catalog is None:
            hier_catalog = []
        if hierarchy <= 1:
            if len(idx) == 0:
                data = torch.ones((3, self.output_size, self.output_size))
                data[0] = data[0] * 77.0 / 255.0  # deep purple
                data[1] = data[1] * 0.0 / 255.0
                data[2] = data[2] * 153.0 / 255.0
                data = torch.swapaxes(data, 0, 2)
                metadata = None
            else:
                vector = healpy.pix2vec(2 ** order, pixel, nest=True)
                distances = numpy.sum(
                    numpy.square(catalog[numpy.array(idx)][:, 4:7] - vector), axis=1
                )
                best = idx[numpy.argmin(distances)]
                data, metadata = dataset[int(catalog[best][0])]
                data = functional.rotate(data, catalog[best][1], expand=False)
                data = functional.center_crop(
                    data, [self.crop_size, self.crop_size]
                )  # crop
                data = self.project_data(data, order, pixel)
            ra, dec = healpy.pix2ang(2 ** order, pixel, lonlat=True, nest=True)
            healpix_mapping = self.make_catalog_row(ra, dec, metadata)
            hier_catalog.append(healpix_mapping)
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
            hier_catalog
        )
        q2 = self.embed_tile(
            dataset,
            catalog,
            order + 1,
            pixel * 4 + 1,
            hierarchy / 2,
            healpix_cells[pixel * 4 + 1],
            hier_catalog
        )
        q3 = self.embed_tile(
            dataset,
            catalog,
            order + 1,
            pixel * 4 + 2,
            hierarchy / 2,
            healpix_cells[pixel * 4 + 2],
            hier_catalog
        )
        q4 = self.embed_tile(
            dataset,
            catalog,
            order + 1,
            pixel * 4 + 3,
            hierarchy / 2,
            healpix_cells[pixel * 4 + 3],
            hier_catalog
        )
        result = torch.ones((q1.shape[0] * 2, q1.shape[1] * 2, 3))
        result[: q1.shape[0], : q1.shape[1]] = q1
        result[q1.shape[0]:, : q1.shape[1]] = q2
        result[: q1.shape[0], q1.shape[1]:] = q3
        result[q1.shape[0]:, q1.shape[1]:] = q4
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

        self.check_folders("interaction_catalog")
        self.create_folders("interaction_catalog")
        self.create_hips_properties("interaction_catalog")

        datamodule.setup("processing")
        dataset = datamodule.data_processing

        catalog = pd.read_csv(
            self.catalog_file,
            usecols=[ "id", "rotation", "x", "y", "z"],
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

            allsky_cat = None
            allsky_path = os.path.join(self.output_folder, self.title, "interaction_catalog", "Norder"+str(i))
            for j in range(12*4**i):
                if j % (int(12*4**i/100)+1) == 0:
                    print(".", end="", flush=True)
                hierarchical_cat = []
                data = self.embed_tile(dataset, catalog, i, j, self.hierarchy, healpix_cells[j],
                                       hier_catalog=hierarchical_cat)
                image = Image.fromarray((numpy.clip(data.detach().numpy(),0,1)*255).astype(numpy.uint8))
                image.save(os.path.join(self.output_folder,
                                        self.title,
                                        "projection",
                                        "Norder"+str(i),
                                        "Dir"+str(int(math.floor(j/10000))*10000),
                                        "Npix"+str(j)+".jpg"))
                if len(hierarchical_cat) > 0:
                    cat_df = pd.DataFrame(data=hierarchical_cat, columns=["main_id", "ra", "dec"])
                    cat_path = os.path.join(allsky_path, "Dir"+str(int(math.floor(j/10000))*10000))
                    with open(os.path.join(cat_path, "Npix"+str(j)+".tsv"), "w", encoding='utf-8') as tsv_cat:
                        tsv_cat.write(cat_df.to_csv(sep='\t'))
                    if allsky_cat is None:
                        allsky_cat = cat_df
                    else:
                        allsky_cat = pd.concat([allsky_cat, cat_df], ignore_index=True)
            with open(os.path.join(allsky_path, "Allsky.tsv"), 'w', encoding='utf-8') as allsky_tsv:
                allsky_tsv.write(allsky_cat.to_csv(sep='\t'))
        if self.verbose > 0:
            print("Generating dataset projection ... done.")

    def make_catalog_row(self, ra, dec, metadata=None):
        dec_str = '+{0}'.format(dec) if dec > 0 else str(dec)
        if metadata is not None:
            main_id = "{simulation}_{snapshot}_{subhalo_id}".format(
                        simulation=metadata["simulation"],
                        snapshot=metadata["snapshot"],
                        subhalo_id=metadata["subhalo_id"])
        else:
            main_id = "undefined"
        healpix_mapping = {
            "main_id": main_id,
            "ra": ra,
            "dec": dec_str
        }
        return healpix_mapping

    def create_images(self, datamodule: SpherinatorDataModule, output_path=None):
        if output_path is None:
            output_path = self.title_folder
        output_path = output_path / Path("morphology")
        output_path.mkdir(parents=True, exist_ok=True)
        datamodule.create_images(output_path)

    def create_gas_pointclouds(self, datamodule, output_path=None):
        if output_path is None:
            output_path = self.title_folder
        output_path = output_path / Path("gasclouds")
        output_path.mkdir(parents=True, exist_ok=True)
        datamodule.create_gas_pointclouds(output_path)

    def create_data_cube(self, datamodule, data_aspects):
        # Todo: We would prefer a data type for this instead of a folder
        base_path = self.title_folder / Path("data_cube")
        for aspect in data_aspects:
            output_path = base_path / Path(aspect)
            output_path.mkdir(parents=True, exist_ok=True)
            create_visuals = getattr(datamodule, f"create_{aspect}")
            create_visuals(output_path)
