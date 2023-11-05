import os
import numpy
import math
import skimage.io as io
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from PIL import Image

data_directory = "/local_data/AIN/Data/HiPSter/IllustrisV12/projection"
extension = ".jpg"
filenames = []
dir = 0
edge_width = 64

if __name__ == "__main__":

    for order in range(4):
        width = math.floor(math.sqrt(12*4**order))
        height = math.ceil(12*4**order/width)
        result = torch.zeros((edge_width*height, edge_width*width, 3))

        print ("order " +str(order)+" - ",end="",flush=True)
        for i in range(12*4**order):
            file = os.path.join(data_directory, "Norder"+str(order), "Dir"+str(dir), "Npix"+str(i)+extension)
            image = torch.swapaxes(torch.Tensor(io.imread(file)), 0, 2) / 255.0
            image = TF.resize(image, [64,64], antialias=False)
            image = torch.swapaxes(image, 0, 2)
            x = i % width
            y = math.floor(i / width)
            result[y*edge_width:(y+1)*edge_width,x*edge_width:(x+1)*edge_width] = image
        image = Image.fromarray((numpy.clip(result.numpy(),0,1)*255).astype(numpy.uint8) , mode="RGB")
        image.save(os.path.join(data_directory, "Norder"+str(order), "Allsky.jpg"))
        print("done!", flush=True)