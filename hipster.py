""" Provides all functionalities to transform a model in a HiPS representation for browsing.
"""

import math
import os
from datetime import datetime
from shutil import rmtree

import healpy
import numpy
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
from PIL import Image
from torch.utils.data import DataLoader

import data.galaxy_zoo_dataset as galaxy_zoo_dataset
import data.illustris_sdss_data_module
import data.illustris_sdss_dataset
import data.preprocessing as preprocessing
from models import RotationalSphericalProjectingAutoencoder


class Hipster():
    """_
    Provides all functions to automatically generate a HiPS representation for a machine learning
    model that projects images on a sphere.
    """

    def __init__(self, output_folder, title, max_order=3, crop_size=64, output_size=128):
        """ Initializes the hipster

        Args:
            output_folder (String): The place where to export the HiPS to. In case it exists, there
                is a user prompt before deleting the folder.
            title (String): The title string to be passed to the meta files.
            max_order (int, optional): The depth of the tiling. Should be smaller than 10.
                Defaults to 3.
            crop_size (int, optional): The size to be cropped from the generating model output, in
                case it might be larger. Defaults to 64.
            output_size (int, optional): Specifies the size the tilings should be scaled to. Must be
                in the powers of 2. Defaults to 128.
        """
        assert math.log2(output_size) == int(math.log2(output_size))
        assert max_order < 10
        self.output_folder = output_folder
        self.title = title
        self.max_order = max_order
        self.crop_size = crop_size
        self.output_size = output_size

    def check_folders(self, base_folder):
        """ Checks whether the base folder exists and deletes it after prompting for user input

        Args:
            base_folder (String): The base folder to check.
        """
        path = os.path.join(self.output_folder, self.title, base_folder)
        if os.path.exists(path):
            answer = input("path "+str(path)+", delete? Yes,[No]")
            if answer == "Yes":
                rmtree(os.path.join(self.output_folder, self.title, base_folder))
            else:
                exit(1)

    def create_folders(self, base_folder):
        """ Creates all folders and sub-folders to store the HiPS tiles.

        Args:
            base_folder (String): The base folder to start the folder creation in.
        """
        print("creating folders:")
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)
        if not os.path.exists(os.path.join(self.output_folder,
                                           self.title)):
            os.mkdir(os.path.join(self.output_folder,
                                  self.title))
        os.mkdir(os.path.join(self.output_folder,
                              self.title,
                              base_folder))
        for i in range(self.max_order+1):
            os.mkdir(os.path.join(self.output_folder,
                                  self.title, base_folder,
                                  "Norder"+str(i)))
            for j in range(int(math.floor(12*4**i/10000))+1):
                os.mkdir(os.path.join(self.output_folder,
                                      self.title,
                                      base_folder,
                                      "Norder"+str(i),
                                      "Dir"+str(j*10000)))

    def create_hips_properties(self, base_folder):
        """ Generates the properties file that contains the meta-information of the HiPS tiling.

        Args:
            base_folder (String): The place where to create the 'properties' file.
        """
        print("creating meta-data:")
        with open(os.path.join(self.output_folder,
                               self.title,
                               base_folder,
                               "properties"), 'w', encoding="utf-8") as output:
            # TODO: add all keywords support and write proper information
            output.write("creator_did          = ivo://HITS/hipster\n")
            output.write("obs_title            = "+self.title+"\n")
            output.write("obs_description      = blablabla\n")
            output.write("dataproduct_type     = image\n")
            output.write("dataproduct_subtype  = color\n")
            output.write("hips_version         = 1.4\n")
            output.write("prov_progenitor      = blablabla\n")
            output.write("hips_creation_date   = "+datetime.now().isoformat()+"\n")
            output.write("hips_release_date    = "+datetime.now().isoformat()+"\n")
            output.write("hips_status          = public master clonable\n")
            output.write("hips_tile_format     = jpeg\n")
            output.write("hips_order           = "+str(self.max_order)+"\n")
            output.write("hips_tile_width      = "+str(self.output_size)+"\n")
            output.write("hips_frame           = equatorial\n")
            output.flush()

    def create_index_file(self, base_folder):
        """ Generates the 'index.html' file that contains an direct access to the HiPS tiling via
            aladin lite.

        Args:
            base_folder (String): The place where to create the 'index.html' file.
        """
        print("creating index.html:")
        with open(os.path.join(self.output_folder,
                               self.title,
                               base_folder,
                               "index.html"), 'w', encoding="utf-8") as output:
            output.write("<!DOCTYPE html>\n")
            output.write("<html>\n")
            output.write("<head>\n")
            output.write("<meta name='description' content='custom HiPS of "+self.title+"'>\n")
            output.write("  <meta charset='utf-8'>\n")
            output.write("  <title>HiPSter representation of "+self.title+"</title>\n")
            output.write("</head>\n")
            output.write("<body>\n")
            output.write("    <div id='aladin-lite-div' style='width:500px;height:500px;'></div>\n")
            output.write("    <script type='text/javascript' " +
                         "src='https://aladin.u-strasbg.fr/AladinLite/api/v3/latest/aladin.js'" +
                         "charset='utf-8'></script>\n")
            output.write("    <script type='text/javascript'>\n")
            output.write("        var aladin;\n")
            output.write("	    A.init.then(() => {\n")
            output.write("            aladin = A.aladin('#aladin-lite-div');\n")
            # TODO: check this current hack for the tile location!!!
            output.write("            aladin.setImageSurvey(aladin.createImageSurvey(" +
                         "'"+self.title+"', " +
                         "'sphere projection of data from"+self.title+"', " +
                         "'http://localhost:8082/"+self.title+"/" +
                         base_folder+"'," +
                         "'equatorial', "+str(self.max_order)+", {imgFormat: 'jpg'})); \n")
            output.write("            aladin.setFoV(180.0); \n")
            output.write("        });\n")
            output.write("    </script>\n")
            output.write("</body>\n")
            output.write("</html>")
            output.flush()

    def generate_hips(self, model):
        """ Generates a HiPS tiling following the standard defined in
            https://www.ivoa.net/documents/HiPS/20170519/REC-HIPS-1.0-20170519.pdf

        Args:
            model (PT.module): A model that allows to call decode(x) for a three dimensional
            vector x. The resulting reconstructions are used to generate the tiles for HiPS.
        """
        self.check_folders("model")
        self.create_folders("model")

        print("creating tiles:")
        for i in range(self.max_order+1):
            print ("  order "+str(i)+" ["+
                   str(12*4**i).rjust(int(math.log10(12*4**self.max_order))+1," ")+" tiles]:",
                   end="")
            for j in range(12*4**i):
                if j % (int(12*4**i/100)+1) == 0:
                    print(".", end="")
                vector = healpy.pix2vec(2**i,j,nest=True)
                vector = torch.tensor(vector).reshape(1,3).type(dtype=torch.float32)
                data = model.decode(vector)[0]
                data = torch.swapaxes(data, 0, 2)
                image = Image.fromarray(
                    (numpy.clip(data.detach().numpy(),0,1)*255).astype(numpy.uint8))
                if image.width != self.output_size or image.height != self.output_size:
                    image = image.resize((self.output_size, self.output_size))
                image.save(os.path.join(self.output_folder,
                                        self.title,
                                        "model",
                                        "Norder"+str(i),
                                        "Dir"+str(int(math.floor(j/10000))*10000),
                                        "Npix"+str(j)+".jpg"))
            print(" done")
        self.create_hips_properties("model")
        self.create_index_file("model")
        print("done!")

    def project_dataset(self, model, dataloader, rotation_steps):
        result_coordinates = torch.zeros((0, 3))
        result_rotations = torch.zeros((0))
        result_losses = torch.zeros((0))
        for batch in dataloader:
            print(".", end="")
            losses = torch.zeros((batch['id'].shape[0],rotation_steps))
            coords = torch.zeros((batch['id'].shape[0],rotation_steps,3))
            images = batch['image']
            for r in range(rotation_steps):
                rot_images = functional.rotate(images, 360/rotation_steps*r, expand=False) # rotate
                crop_images = functional.center_crop(rot_images, [256,256]) # crop
                scaled_images = functional.resize(crop_images, [64,64], antialias=False) # scale
                with torch.no_grad():
                    reconstruction, coordinates = model.forward(scaled_images)
                    losses[:,r] = model.spherical_loss(scaled_images, reconstruction, coordinates)
                    coords[:,r] = model.scale_to_unity(coordinates)
            min = torch.argmin(losses, dim=1)
            result_coordinates = torch.cat((result_coordinates, coords[torch.arange(batch['id'].shape[0]),min]))
            result_rotations = torch.cat((result_rotations, 360.0/rotation_steps*min))
            result_losses = torch.cat((result_losses, losses[torch.arange(batch['id'].shape[0]),min]))
        return result_coordinates, result_rotations, result_losses

    def generate_catalog(self, model, dataloader, catalog_file):
        """ Generates a catalog by mapping all provided data using the encoder of the
            trained model. The catalog will contain the id, coordinates in angles as well
            as unity vector coordinates, the reconstruction loss, and a link to the original
            image file.

        Args:
            model (PT.module): A model that allows to call project_dataset(x) to encode elements
                of a dataset to a sphere.
            dataloader (DataLoader): A data loader to access the images that should get mapped.
            catalog_file (String): Name of the csv file to be generated.
        """
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)
        if not os.path.exists(os.path.join(self.output_folder,
                                           self.title)):
            os.mkdir(os.path.join(self.output_folder,
                                  self.title))
        if os.path.exists(os.path.join(self.output_folder,
                                       self.title,
                                       catalog_file)): #delete existing catalog
            answer = input("catalog exists, overwrite? Yes,[No]")
            if answer != "Yes":
                return
        print("projecting dataset:")
        coordinates, rotations, losses = self.project_dataset(model, dataloader, 36)
        coordinates = coordinates.cpu().detach().numpy()
        rotations = rotations.cpu().detach().numpy()
        angles = numpy.array(healpy.vec2ang(coordinates))*180.0/math.pi
        angles = angles.T

        print("creating catalog file:")
        with open(os.path.join(self.output_folder,
                               self.title,
                               "catalog.csv"), 'w', encoding="utf-8") as output:
            output.write("#id,RA2000,DEC2000,rotation,x,y,z,loss,filename\n")
            for i in range(coordinates.shape[0]):
                output.write(str(i)+","+str(angles[i,1])+"," +
                             str(90.0-angles[i,0])+"," +
                             str(rotations[i])+",")
                output.write(str(coordinates[i,0])+"," +
                             str(coordinates[i,1])+"," +
                             str(coordinates[i,2])+",")
                output.write(str(losses[i])+",")
                output.write("http://localhost:8083" +
                             dataloader.dataset[i]['filename']+"\n")
            output.flush()
        print("done!")

    def generate_dataset_projection(self, dataset, catalog_file):
        """ Generates a HiPS tiling by using the coordinates of every image to map the original
            images form the data set based on their distance to the closest heal pixel cell
            center.

        Args:
            dataset (Dataset): The dataset to access the original images
            catalog_file (String): The previously created catalog file that contains the
                coordinates.
        """
        self.check_folders("projection")
        self.create_folders("projection")

        print("reading catalog")
        catalog = numpy.genfromtxt(os.path.join(self.output_folder,
                                                self.title,
                                                catalog_file),
                                   delimiter=",",
                                   skip_header=1,
                                   usecols=[0,1,2,3,4,5,6])

        print("creating tiles:")
        for i in range(self.max_order+1):
            healpix_cells = {} # create an extra map to quickly find images in a cell
            for j in range(12*4**i):
                healpix_cells[j] = []
            for element, number in zip(catalog, range(catalog.shape[0])):
                healpix_cells[healpy.vec2pix(2**i,
                                             element[4],
                                             element[5],
                                             element[6],
                                             nest=True)].append(int(number))

            print ("\n  order "+str(i)+" [" +
                   str(12*4**i).rjust(int(math.log10(12*4**self.max_order))+1," ")+" tiles]:",
                   end="")
            for j in range(12*4**i):
                if j % (int(12*4**i/100)+1) == 0:
                    print(".", end="")
                vector = healpy.pix2vec(2**i,j,nest=True)
                if len(healpix_cells[j]) == 0:
                    data = torch.ones((3,self.output_size,self.output_size))
                    data[1] = torch.zeros((self.output_size,self.output_size))
                else:
                    distances = numpy.sum(numpy.square(
                        catalog[numpy.array(healpix_cells[j])][:,4:7] - vector), axis=1)
                    best = healpix_cells[j][numpy.argmin(distances)]
                    data = dataset[int(catalog[best][0])]['image']
                    data = functional.rotate(data, catalog[best][3], expand=False)
                data = torch.swapaxes(data, 0, 2)
                image = Image.fromarray((
                    numpy.clip(data.detach().numpy(),0,1)*255).astype(numpy.uint8))
                if image.width > self.crop_size or image.height > self.crop_size:
                    image = image.crop((int(image.width/2-self.crop_size/2),
                                        int(image.height/2-self.crop_size/2),
                                        int(image.width/2+self.crop_size/2),
                                        int(image.height/2+self.crop_size/2)))
                if image.width != self.output_size or image.height != self.output_size:
                    image = image.resize((self.output_size, self.output_size))
                image.save(os.path.join(self.output_folder,
                                        self.title,
                                        "projection",
                                        "Norder"+str(i),
                                        "Dir"+str(int(math.floor(j/10000))*10000),
                                        "Npix"+str(j)+".jpg"))

        self.create_hips_properties("projection")
        self.create_index_file("projection")
        print("done!")

if __name__ == "__main__":
    myHipster = Hipster("/local_data/AIN/Data/HiPSter", "Illustris", max_order=5, crop_size=256, output_size=256)
    myModel = RotationalSphericalProjectingAutoencoder()
    checkpoint = torch.load("illustris.epoch908step64539.ckpt")
    myModel.load_state_dict(checkpoint["state_dict"])

    #myHipster.generate_hips(myModel)

    # myDataset = galaxy_zoo_dataset.GalaxyZooDataset(data_directory="/hits/basement/ain/Data/KaggleGalaxyZoo/images_training_rev1",#efigi-1.6/png"
    #                                     extension=".jpg",
    #                                     transform = transforms.Compose([#Preprocessing.DielemanTransformation(rotation_range=[0], translation_range=[4./424,4./424], scaling_range=[1/1.1,1.1], flip=0.5),
    #                                                                    #Preprocessing.KrizhevskyColorTransformation(weights=[-0.0148366, -0.01253134, -0.01040762], std=0.5),
    #                                                                     preprocessing.CropAndScale((424,424), (424,424))
    #                                                                    ])
    #                                    )

    # myDataloader = DataLoader(myDataset, batch_size=1024, shuffle=False, num_workers=16)

    myDataModule = data.illustris_sdss_data_module.IllustrisSdssDataModule(
                      ["/local_data/AIN/SKIRT_synthetic_images/TNG100/sdss/snapnum_099/data/",
                       "/local_data/AIN/SKIRT_synthetic_images/TNG100/sdss/snapnum_095/data/",
                       "/local_data/AIN/SKIRT_synthetic_images/TNG50/sdss/snapnum_099/data/",
                       "/local_data/AIN/SKIRT_synthetic_images/TNG50/sdss/snapnum_095/data/",
                       "/local_data/AIN/SKIRT_synthetic_images/Illustris/sdss/snapnum_135/data/",
                       "/local_data/AIN/SKIRT_synthetic_images/Illustris/sdss/snapnum_131/data/"],
                      extension=".fits",
                      minsize=100,
                      shuffle=False,
                      batch_size=512,
                      num_workers=32)
    myDataModule.setup("val")

    myHipster.generate_catalog(myModel, myDataModule.val_dataloader(), "catalog.csv")

    myHipster.generate_dataset_projection(myDataModule.data_val, "catalog.csv")

    #TODO: currently you manually have to call 'python3 -m http.server 8082' to start a simple web server providing access to the tiles.
