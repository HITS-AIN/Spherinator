#!/usr/bin/env python3

""" Provides all functionalities to transform a model in a HiPS representation for browsing.
"""

import argparse
import importlib
import math
import os
from datetime import datetime
from shutil import rmtree

import healpy
import numpy
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
import yaml
from astropy.io.votable import writeto
from astropy.table import Table
from PIL import Image


class Hipster():
    """_
    Provides all functions to automatically generate a HiPS representation for a machine learning
    model that projects images on a sphere.
    """

    def __init__(self, output_folder, title, max_order=3, hierarchy=1, crop_size=64, output_size=128):
        """ Initializes the hipster

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
        """
        assert math.log2(output_size) == int(math.log2(output_size))
        assert max_order < 10
        self.output_folder = output_folder
        self.title = title
        self.max_order = max_order
        self.hierarchy = hierarchy
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
            output.write("hips_tile_width      = "+str(self.output_size*self.hierarchy)+"\n")
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

    def generate_tile(self, model, order, pixel, hierarchy):
        if hierarchy<=1:
            vector = healpy.pix2vec(2**order,pixel,nest=True)
            vector = torch.tensor(vector).reshape(1,3).type(dtype=torch.float32)
            data = model.reconstruct(vector)[0]
            data = functional.resize(data, [self.output_size,self.output_size], antialias=False)
            data = torch.swapaxes(data, 0, 2)
            return data
        q1 = self.generate_tile(model, order+1, pixel*4, hierarchy/2)
        q2 = self.generate_tile(model, order+1, pixel*4+1, hierarchy/2)
        q3 = self.generate_tile(model, order+1, pixel*4+2, hierarchy/2)
        q4 = self.generate_tile(model, order+1, pixel*4+3, hierarchy/2)
        result = torch.ones((q1.shape[0]*2, q1.shape[1]*2,3))
        result[:q1.shape[0],:q1.shape[1]] = q1
        result[q1.shape[0]:,:q1.shape[1]] = q2
        result[:q1.shape[0],q1.shape[1]:] = q3
        result[q1.shape[0]:,q1.shape[1]:] = q4
        return result

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
                    print(".", end="", flush=True)
                image = self.generate_tile(model, i, j, self.hierarchy)
                image = Image.fromarray((numpy.clip(image.detach().numpy(),0,1)*255).astype(numpy.uint8))
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

    def transform_csv_to_votable(self, csv_filename, votable_filename):
        input_file = os.path.join(self.output_folder,
                             self.title,
                             csv_filename)
        output_file = os.path.join(self.output_folder,
                              self.title,
                              votable_filename)
        table = Table.read(input_file, format='ascii.csv')
        writeto(table, output_file)

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
                scaled_images = functional.resize(crop_images, [128,128], antialias=False) # scale
                with torch.no_grad():
                    coordinates = model.project(scaled_images)
                    reconstruction = model.reconstruct(coordinates)
                    losses[:,r] = model.reconstruction_loss(scaled_images, reconstruction)
                    coords[:,r] = coordinates
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
        losses = losses.cpu().detach().numpy()
        angles = numpy.array(healpy.vec2ang(coordinates))*180.0/math.pi
        angles = angles.T

        print("creating catalog file:")
        with open(os.path.join(self.output_folder,
                               self.title,
                               "catalog.csv"), 'w', encoding="utf-8") as output:
            output.write("#preview,simulation,snapshot data,subhalo id,subhalo,RMSE,id,RA2000,DEC2000,rotation,x,y,z\n")
            for i in range(coordinates.shape[0]):
                output.write("<a href='https://space.h-its.org/Illustris/jpg/")
                output.write(str(dataloader.dataset[i]['metadata']['simulation'])+"/")
                output.write(str(dataloader.dataset[i]['metadata']['snapshot'])+"/")
                output.write(str(dataloader.dataset[i]['metadata']['subhalo_id'])+".jpg' target='_blank'>")
                output.write("<img src='https://space.h-its.org/Illustris/thumbnails/")
                output.write(str(dataloader.dataset[i]['metadata']['simulation'])+"/")
                output.write(str(dataloader.dataset[i]['metadata']['snapshot'])+"/")
                output.write(str(dataloader.dataset[i]['metadata']['subhalo_id'])+".jpg'></a>,")

                output.write(str(dataloader.dataset[i]['metadata']['simulation'])+",")
                output.write(str(dataloader.dataset[i]['metadata']['snapshot'])+",")
                output.write(str(dataloader.dataset[i]['metadata']['subhalo_id'])+",")
                output.write("<a href='")
                output.write("https://www.illustris-project.org/api/")
                output.write(str(dataloader.dataset[i]['metadata']['simulation'])+"-1/snapshots/")
                output.write(str(dataloader.dataset[i]['metadata']['snapshot'])+"/subhalos/")
                output.write(str(dataloader.dataset[i]['metadata']['subhalo_id'])+"/")
                output.write("' target='_blank'>www.illustris-project.org</a>,")
                output.write(str(losses[i])+",")
                output.write(str(i)+","+str(angles[i,1])+"," +
                             str(90.0-angles[i,0])+"," +
                             str(rotations[i])+",")
                output.write(str(coordinates[i,0])+"," +
                             str(coordinates[i,1])+"," +
                             str(coordinates[i,2])+"\n")


            output.flush()
        print("done!")

    def calculate_healpix_cells(self, catalog, numbers, order, pixels):
        healpix_cells = {} # create an extra map to quickly find images in a cell
        for pixel in pixels:
            healpix_cells[pixel] = [] # create empty lists for each cell
        for number in numbers:
            pixel = healpy.vec2pix(2**order,
                                   catalog[number][4],
                                   catalog[number][5],
                                   catalog[number][6],
                                   nest=True)
            if pixel in healpix_cells:
                healpix_cells[pixel].append(int(number))
        return healpix_cells

    def embed_tile(self, dataset, catalog, order, pixel, hierarchy, idx):
        if hierarchy <= 1:
            if len(idx) == 0:
                data = torch.ones((3,self.output_size,self.output_size))
                data[0] = data[0]*77.0/255.0 # deep purple
                data[1] = data[1]*0.0/255.0
                data[2] = data[2]*153.0/255.0
            else:
                vector = healpy.pix2vec(2**order,pixel,nest=True)
                distances = numpy.sum(numpy.square(
                     catalog[numpy.array(idx)][:,4:7] - vector), axis=1)
                best = idx[numpy.argmin(distances)]
                data = dataset[int(catalog[best][0])]['image']
                data = functional.rotate(data, catalog[best][3], expand=False)
                data = functional.center_crop(data, [self.crop_size,self.crop_size]) # crop
                data = functional.resize(data, [self.output_size,self.output_size], antialias=False) # scale
            data = torch.swapaxes(data, 0, 2)
            return data
        healpix_cells = self.calculate_healpix_cells(catalog, idx, order+1, range(pixel*4,pixel*4+4))
        q1 = self.embed_tile(dataset, catalog, order+1, pixel*4, hierarchy/2, healpix_cells[pixel*4])
        q2 = self.embed_tile(dataset, catalog, order+1, pixel*4+1, hierarchy/2, healpix_cells[pixel*4+1])
        q3 = self.embed_tile(dataset, catalog, order+1, pixel*4+2, hierarchy/2, healpix_cells[pixel*4+2])
        q4 = self.embed_tile(dataset, catalog, order+1, pixel*4+3, hierarchy/2, healpix_cells[pixel*4+3])
        result = torch.ones((q1.shape[0]*2, q1.shape[1]*2,3))
        result[:q1.shape[0],:q1.shape[1]] = q1
        result[q1.shape[0]:,:q1.shape[1]] = q2
        result[:q1.shape[0],q1.shape[1]:] = q3
        result[q1.shape[0]:,q1.shape[1]:] = q4
        return result

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
                                   usecols=[6,7,8,9,10,11,12]) ##id,RA2000,DEC2000,rotation,x,y,z

        print("creating tiles:")
        for i in range(self.max_order+1):
            healpix_cells = self.calculate_healpix_cells(catalog, range(catalog.shape[0]), i, range(12*4**i))
            print ("\n  order "+str(i)+" [" +
                   str(12*4**i).rjust(int(math.log10(12*4**self.max_order))+1," ")+" tiles]:",
                   end="")
            for j in range(12*4**i):
                if j % (int(12*4**i/100)+1) == 0:
                    print(".", end="", flush=True)
                data = self.embed_tile(dataset, catalog, i, j, self.hierarchy, healpix_cells[j])
                image = Image.fromarray((
                    numpy.clip(data.detach().numpy(),0,1)*255).astype(numpy.uint8))
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

    parser = argparse.ArgumentParser(description="Transform a model in a HiPS representation")
    parser.add_argument("task", help="Execution task [hips, catalog, projection, all].")
    parser.add_argument("--config", "-c", default="config.yaml",
                        help="config file (default = 'config.yaml').")
    parser.add_argument("--checkpoint", "-m", default="model.ckpt",
                        help="checkpoint file (default = 'model.ckpt').")
    parser.add_argument("--max_order", default=4, type=int,
                        help="Maximal order of HiPS tiles (default = 4).")
    parser.add_argument("--hierarchy", default=8, type=int,
                        help="Maximal order of HiPS tiles (default = 8).")
    parser.add_argument("--crop_size", default=256, type=int,
                        help="Image crop size (default = 256).")
    parser.add_argument("--output_size", default=256, type=int,
                        help="Image output size (default = 64).")
    parser.add_argument("--output_folder", default='./HiPSter',
                        help="Output of HiPS (default = './HiPSter').")
    parser.add_argument("--title", default='IllustrisV2',
                        help="HiPS title (default = 'IllustrisV2').")

    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)

    # Import the model class and create an instance of it
    if args.task in ["hips", "catalog", "all"]:
        model_class_path = config['model']['class_path']
        module_name, class_name = model_class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
        model_init_args = config['model']['init_args']
        myModel = model_class(**model_init_args)

        checkpoint = torch.load(args.checkpoint)
        myModel.load_state_dict(checkpoint["state_dict"])

    # Import the data module and create an instance of it
    if args.task in ["catalog", "projection", "all"]:
        data_class_path = config['data']['class_path']
        module_name, class_name = data_class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        data_class = getattr(module, class_name)
        data_init_args = config['data']['init_args']
        myDataModule = data_class(**data_init_args)
        myDataModule.setup("predict")

    myHipster = Hipster(args.output_folder, args.title,
                        max_order=args.max_order, hierarchy=args.hierarchy,
                        crop_size=args.crop_size, output_size=args.output_size)

    if (args.task == "hips" or args.task == "all"):
        myHipster.generate_hips(myModel)

    if (args.task == "catalog" or args.task == "all"):
        myHipster.generate_catalog(myModel, myDataModule.predict_dataloader(), "catalog.csv")
        myHipster.transform_csv_to_votable("catalog.csv", "catalog.vot")

    if (args.task == "projection" or args.task == "all"):
        myHipster.generate_dataset_projection(myDataModule.data_predict, "catalog.csv")

    #TODO: currently you manually have to call 'python3 -m http.server 8082' to start a simple web server providing access to the tiles.
