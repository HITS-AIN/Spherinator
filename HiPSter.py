import math
import os
from datetime import datetime
from shutil import rmtree

import healpy
import numpy
import torch
from PIL import Image

from models import RotationalSphericalProjectingAutoencoder


class HiPSter():
    def __init__(self, output_folder, title, max_order=3, crop_size=64, output_size=128):
        self.output_folder = output_folder
        self.title = title
        self.max_order = max_order
        self.crop_size = crop_size
        self.output_size = output_size
    
    def generate_HiPS(self, model):
        """
        generates a HiPS tiling following https://www.ivoa.net/documents/HiPS/20170519/REC-HIPS-1.0-20170519.pdf
        """
        if os.path.exists(self.output_folder): #delete existing folder
            answer = input("path exists, delete? Yes,[No]")
            if answer == "Yes":
                rmtree(self.output_folder)
        
        print("creating folders:")
        os.mkdir(self.output_folder)
        for i in range(self.max_order+1):
            os.mkdir(os.path.join(self.output_folder, "Norder"+str(i)))
            for j in range(int(math.floor(12*4**i/10000))+1):
                os.mkdir(os.path.join(self.output_folder, "Norder"+str(i), "Dir"+str(j*10000)))
        print("creating tiles:")
        for i in range(self.max_order+1):
            print ("\n  order "+str(i)+" ["+str(12*4**i).rjust(int(math.log10(12*4**self.max_order))+1," ")+" tiles]:", end="")
            for j in range(12*4**i):
                if j % (int(12*4**i/100)+1) == 0:
                    print(".", end="")
                vector = healpy.pix2vec(2**i,j,nest=True)
                vector = torch.tensor(vector).reshape(1,3).type(dtype=torch.float32)
                #vector[:, [0,2]] = vector[:, [2,0]]
                data = model.decode(vector)[0]
                data = torch.swapaxes(data, 0, 2)
                image = Image.fromarray((numpy.clip(data.detach().numpy(),0,1)*255).astype(numpy.uint8))
                #image = image.crop((image.width/2-self.crop_size/2,image.height/2-self.crop_size/2,image.width/2+self.crop_size/2,image.height/2+self.crop_size/2))
                image = image.resize((self.output_size, self.output_size))
                image.save(os.path.join(self.output_folder, "Norder"+str(i), "Dir"+str(int(math.floor(j/10000))*10000), "Npix"+str(j)+".jpg"))
        print("\ncreating meta-data:")
        with open(os.path.join(self.output_folder, "properties"), 'w') as f:
            f.write("creator_did          = ivo://HITS/hipster\n")
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
        with open(os.path.join(self.output_folder, "index.html"), 'w') as f:
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
            f.write("            aladin.setImageSurvey(aladin.createImageSurvey('http://localhost:8082/"+self.output_folder+"', 'sphere projection of data from"+self.title+"', 'http://localhost:8082/"+self.output_folder+"', 'equatorial', "+str(self.max_order)+", {imgFormat: 'jpg'})); \n")
            f.write("            aladin.setFoV(180.0); \n")
            f.write("        });\n")
            f.write("    </script>\n")
            f.write("</body>\n")
            f.write("</html>")
            f.flush()
        print("done!")
        
if __name__ == "__main__":
    hipster = HiPSter("HiPSter", "GZ", max_order=4, crop_size=64, output_size=64)
    model = train.RotationalSphericalProjectingAutoencoder()
    checkpoint = torch.load("epoch=14-step=4185.ckpt")
    model.load_state_dict(checkpoint["state_dict"])
    hipster.generate_HiPS(model)