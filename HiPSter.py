import math
import os
from datetime import datetime
from shutil import rmtree

import healpy
import numpy
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader

import DataSets
import Preprocessing
from models import RotationalSphericalProjectingAutoencoder


class HiPSter():
    """_
    Provides all functions to automatically generate a HiPS representation for a machine learning model
    that projects images on a sphere.
    """

    def __init__(self, output_folder, title, max_order=3, crop_size=64, output_size=128):
        """Initializes the HiPSter

        Args:
            output_folder (String): The place where to export the HiPS to. In case it exists, there is a user prompt before deleting the folder.
            title (String): The title string to be passed to the meta files.
            max_order (int, optional): The depth of the tiling. Should be smaller than 10. Defaults to 3.
            crop_size (int, optional): The size to be cropped from the generating model output, in case it might be larger. Defaults to 64.
            output_size (int, optional): Specifies the size the tilings should be scaled to. Must be in the powers of 2. Defaults to 128.
        """
        assert math.log2(output_size) == int(math.log2(output_size))
        assert max_order < 10
        self.output_folder = output_folder
        self.title = title
        self.max_order = max_order
        self.crop_size = crop_size
        self.output_size = output_size

    def check_folders(self, base_folder):
        if os.path.exists(os.path.join(self.output_folder, self.title, base_folder)): #delete existing folder
            answer = input("path exists, delete? Yes,[No]")
            if answer == "Yes":
                rmtree(os.path.join(self.output_folder, self.title, base_folder))
            else:
                exit(1)

    def create_folders(self, base_folder):
        print("creating folders:")
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)
        if not os.path.exists(os.path.join(self.output_folder, self.title)):
            os.mkdir(os.path.join(self.output_folder, self.title))
        os.mkdir(os.path.join(self.output_folder, self.title, base_folder))
        for i in range(self.max_order+1):
            os.mkdir(os.path.join(self.output_folder, self.title, base_folder, "Norder"+str(i)))
            for j in range(int(math.floor(12*4**i/10000))+1):
                os.mkdir(os.path.join(self.output_folder, self.title, base_folder, "Norder"+str(i), "Dir"+str(j*10000)))

    def create_HiPS_properties(self, base_folder):
        print("creating meta-data:")
        with open(os.path.join(self.output_folder, self.title, base_folder, "properties"), 'w') as f: # create the properties file
            f.write("creator_did          = ivo://HITS/hipster\n") # TODO: add all keywords support and write proper information
            f.write("obs_title            = "+self.title+"\n")
            f.write("obs_description      = blablabla\n")
            f.write("dataproduct_type     = image\n")
            f.write("dataproduct_subtype  = color\n")
            f.write("hips_version         = 1.4\n")
            f.write("prov_progenitor      = blablabla\n")
            f.write("hips_creation_date   = "+datetime.now().isoformat()+"\n")
            f.write("hips_release_date    = "+datetime.now().isoformat()+"\n")
            f.write("hips_status          = public master clonable\n")
            f.write("hips_tile_format     = jpeg\n")
            f.write("hips_order           = "+str(self.max_order)+"\n")
            f.write("hips_tile_width      = "+str(self.output_size)+"\n")
            f.write("hips_frame           = equatorial\n")
            f.flush()

    def create_index_file(self, base_folder):
        print("creating index.html:")
        with open(os.path.join(self.output_folder, self.title, base_folder, "index.html"), 'w') as f: # create the html file to start aladin lite
            f.write("<!DOCTYPE html>\n")
            f.write("<html>\n")
            f.write("<head>\n")
            f.write("<meta name='description' content='custom HiPS of "+self.title+"'>\n")
            f.write("  <meta charset='utf-8'>\n")
            f.write("  <title>HiPSter representation of "+self.title+"</title>\n")
            f.write("</head>\n")
            f.write("<body>\n")
            f.write("    <div id='aladin-lite-div' style='width:500px;height:500px;'></div>\n")
            f.write("    <script type='text/javascript' src='https://aladin.u-strasbg.fr/AladinLite/api/v3/latest/aladin.js' charset='utf-8'></script>\n")
            f.write("    <script type='text/javascript'>\n")
            f.write("        var aladin;\n")
            f.write("	    A.init.then(() => {\n")
            f.write("            aladin = A.aladin('#aladin-lite-div');\n")
            # TODO: check this current hack for the tile location!!!
            f.write("            aladin.setImageSurvey(aladin.createImageSurvey('http://localhost:8082/"+self.output_folder+"/"+self.title+"/"+base_folder+"', 'sphere projection of data from"+self.title+"', 'http://localhost:8082/"+self.output_folder+"/"+self.title+"/"+base_folder+"', 'equatorial', "+str(self.max_order)+", {imgFormat: 'jpg'})); \n")
            f.write("            aladin.setFoV(180.0); \n")
            f.write("        });\n")
            f.write("    </script>\n")
            f.write("</body>\n")
            f.write("</html>")
            f.flush()

    def generate_HiPS(self, model):
        """Generates a HiPS tiling following https://www.ivoa.net/documents/HiPS/20170519/REC-HIPS-1.0-20170519.pdf

        Args:
            model (PT.module): A model that allows to call decode(x) for a three dimensional vector x. The resulting reconstructions are used to generate the tiles for HiPS.
        """
        self.check_folders("model")
        self.create_folders("model")

        print("creating tiles:")
        for i in range(self.max_order+1):
            print ("\n  order "+str(i)+" ["+str(12*4**i).rjust(int(math.log10(12*4**self.max_order))+1," ")+" tiles]:", end="")
            for j in range(12*4**i):
                if j % (int(12*4**i/100)+1) == 0:
                    print(".", end="")
                vector = healpy.pix2vec(2**i,j,nest=True)
                vector = torch.tensor(vector).reshape(1,3).type(dtype=torch.float32)
                data = model.decode(vector)[0]
                data = torch.swapaxes(data, 0, 2)
                image = Image.fromarray((numpy.clip(data.detach().numpy(),0,1)*255).astype(numpy.uint8))
                if image.width > self.crop_size or image.height > self.crop_size: # do a center crop
                    image = image.crop((int(image.width/2-self.crop_size/2),int(image.height/2-self.crop_size/2),
                                        int(image.width/2+self.crop_size/2),int(image.height/2+self.crop_size/2)))
                if image.width != self.output_size or image.height != self.output_size: # rescale to output resolution
                    image = image.resize((self.output_size, self.output_size))
                image.save(os.path.join(self.output_folder, self.title, "model", "Norder"+str(i), "Dir"+str(int(math.floor(j/10000))*10000), "Npix"+str(j)+".jpg"))
            print(" done")
        self.create_HiPS_properties("model")
        self.create_index_file("model")
        print("done!")

    def generate_catalog(self, model, dataloader, catalog_file):
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)
        if not os.path.exists(os.path.join(self.output_folder, self.title)):
            os.mkdir(os.path.join(self.output_folder, self.title))
        if os.path.exists(os.path.join(self.output_folder, self.title, catalog_file)): #delete existing catalog
            answer = input("catalog exists, delete? Yes,[No]")
            if answer != "Yes":
                return
        print("projecting dataset:")
        coordinates, rotations = model.project_dataset(dataloader, 36)
        coordinates = coordinates.cpu().detach().numpy()
        rotations = rotations.cpu().detach().numpy()
        angles = numpy.array(healpy.vec2ang(coordinates))*180.0/math.pi
        angles = angles.T

        print("creating catalog file:")
        with open(os.path.join(self.output_folder, self.title, "catalog.csv"), 'w') as f:
            f.write("#id,RA2000,DEC2000,rotation,x,y,z,pix3,pix4,filename\n")
            for i in range(coordinates.shape[0]):
                f.write(str(i)+","+str(angles[i,1])+","+str(90.0-angles[i,0])+","+str(rotations[i])+",")
                f.write(str(coordinates[i,0])+","+str(coordinates[i,1])+","+str(coordinates[i,2])+",")
                f.write(str(healpy.vec2pix(2**3, coordinates[i,0], coordinates[i,1], coordinates[i,2], nest=True))+",")
                f.write(str(healpy.vec2pix(2**4, coordinates[i,0], coordinates[i,1], coordinates[i,2], nest=True))+",")
                f.write("http://localhost:8083"+dataset.__getitem__(i)['filename']+"\n")
            f.flush()
        print("done!")

    def generate_dataset_projection(self, dataset, catalog_file):
        self.check_folders("projection")
        self.create_folders("projection")

        print("reading catalog")
        catalog = numpy.genfromtxt(os.path.join(self.output_folder, self.title, catalog_file), delimiter=",", skip_header=1, usecols=[0,1,2,3,4,5,6])

        print("creating tiles:")
        for i in range(self.max_order+1):
            healpix_cells = {} # create an extra map to quickly find images in a cell for a given order
            for j in range(12*4**i):
                healpix_cells[j] = []
            for element in catalog:
                healpix_cells[healpy.vec2pix(2**i, element[4], element[5], element[6], nest=True)].append(int(element[0]))

            print ("\n  order "+str(i)+" ["+str(12*4**i).rjust(int(math.log10(12*4**self.max_order))+1," ")+" tiles]:", end="")
            for j in range(12*4**i):
                if j % (int(12*4**i/100)+1) == 0:
                    print(".", end="")
                vector = healpy.pix2vec(2**i,j,nest=True)
                if len(healpix_cells[j]) == 0:
                    data = torch.ones((3,self.output_size,self.output_size))
                    data[1] = torch.zeros((self.output_size,self.output_size))
                else:
                    distances = numpy.sum(numpy.square(catalog[numpy.array(healpix_cells[j])][:,4:7] - vector), axis=1)
                    data = dataset.__getitem__(healpix_cells[j][numpy.argmin(distances)])['image']
                data = torch.swapaxes(data, 0, 2)
                image = Image.fromarray((numpy.clip(data.detach().numpy(),0,1)*255).astype(numpy.uint8))
                if image.width > self.crop_size or image.height > self.crop_size: # do a center crop
                    image = image.crop((int(image.width/2-self.crop_size/2),int(image.height/2-self.crop_size/2),
                                        int(image.width/2+self.crop_size/2),int(image.height/2+self.crop_size/2)))
                if image.width != self.output_size or image.height != self.output_size: # rescale to output resolution
                    image = image.resize((self.output_size, self.output_size))
                image.save(os.path.join(self.output_folder, self.title, "projection", "Norder"+str(i), "Dir"+str(int(math.floor(j/10000))*10000), "Npix"+str(j)+".jpg"))

        self.create_HiPS_properties("projection")
        self.create_index_file("projection")
        print("done!")

if __name__ == "__main__":
    hipster = HiPSter("HiPSter", "GZ", max_order=5, crop_size=64, output_size=64)
    model = RotationalSphericalProjectingAutoencoder()
    #checkpoint = torch.load("efigi_epoch2148-step150430.ckpt")
    checkpoint = torch.load("gz_epoch4523-step1090284.ckpt")
    model.load_state_dict(checkpoint["state_dict"])

    #hipster.generate_HiPS(model)

    dataset = DataSets.GalaxyZooDataset(data_directory="/hits/basement/ain/Data/KaggleGalaxyZoo/images_training_rev1",#efigi-1.6/png",#KaggleGalaxyZoo/images_training_rev1",
                                        extension=".jpg",
                                        transform = transforms.Compose([#Preprocessing.DielemanTransformation(rotation_range=[0], translation_range=[4./424,4./424], scaling_range=[1/1.1,1.1], flip=0.5),
                                                                       #Preprocessing.KrizhevskyColorTransformation(weights=[-0.0148366, -0.01253134, -0.01040762], std=0.5),
                                                                        Preprocessing.CropAndScale((424,424), (424,424))
                                                                       ])
                                       )

    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=16)

    #hipster.generate_catalog(model, dataloader, "catalog.csv")

    hipster.generate_dataset_projection(dataset, "catalog.csv")

    #TODO: currently you manually have to call 'python3 -m http.server 8082' to start a simple web server providing access to the tiles.