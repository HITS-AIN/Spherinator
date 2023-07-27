import numpy
import copy

import torch
import torchvision.transforms.functional as TF
from PIL import Image
import torchvision.transforms as transforms

# FIXME: implement this
class DielemanTransformation():
    def __init__(self, rotation_range, translation_range, scaling_range, flip):
        self.scaling_range = scaling_range
        self.random_affine = transforms.RandomAffine(degrees=rotation_range, translate=translation_range, shear=None)
        self.flip = transforms.RandomHorizontalFlip(p=flip)

    def __call__(self, x):
        input_image = x['image']
        transformed_image = self.random_affine.__call__(input_image)
        zoom = numpy.exp(numpy.random.uniform(numpy.log(self.scaling_range[0]), numpy.log(self.scaling_range[1])))
        resize = TF.resize(transformed_image, (int(input_image.shape[1]*zoom),int(input_image.shape[2]*zoom)), antialias=True)
        x['image'] = self.flip.__call__(resize)
        return x

# TODO: add documentation
class KrizhevskyColorTransformation():
    def __init__(self, weights, std):
        self.weights = torch.tensor(numpy.array(weights))
        self.std = std

    def __call__(self, x):
        """
        Performs a random brightness scaling based on the specied weights and standard deviation following the PCA based idea by Alex Krizhevsky et al. 2012 as used by Dieleman et al. 2015.
        
        Parameters
        ----------
        x : dictonary
            A dictonary conatining the data items returned from the data set. They key 'image' refers to an array of RGB values in the range between 0 and 1.
        
        Returns
        -------
        dictonary
            modifies the entered dictonary and changes the 'image' array accordingly. Ensures that the returned values of the image are between 0 and 1
        """
        transformed_image = x['image']
        noise = torch.normal(0.0, self.std, size=[1]) * self.weights
        transformed_image[0] = transformed_image[0] + noise[0]
        transformed_image[1] = transformed_image[1] + noise[1]
        transformed_image[2] = transformed_image[2] + noise[2]
        transformed_image = torch.clip(transformed_image, 0, 1)
        x['image'] = transformed_image
        return x

class CropAndScale():
    def __init__(self, crop_size, scale_size):
        self.crop_size = crop_size
        self.scale_size = scale_size
    
    def __call__(self, x):
        transformed_image = x['image']
        crop = TF.center_crop(transformed_image, self.crop_size)
        resize = TF.resize(crop, self.scale_size, antialias=False)
        x['image'] = resize
        return x



# TODO: document this
class ViewpointTransformation():
    ROTATIONS = [0, 90, 270, 180]

    def __init__(self, target_size, crop_size, downsampling_factor, rotation_angles=[0], add_flipped_viewport=False):
        self.target_size = target_size
        self.crop_size = crop_size
        self.downsampling_factor = downsampling_factor
        self.rotation_angles = rotation_angles
        self.add_flipped_viewport = add_flipped_viewport

    def __call__(self, x):
        result = torch.zeros((4*len(self.rotation_angles)*(2 if self.add_flipped_viewport else 1),3,self.target_size[0],self.target_size[1]))
        transformed_image = x['image']
        n = 0
        for angle in self.rotation_angles:
            rotation  = TF.rotate(transformed_image, angle)
            crop = TF.center_crop(rotation, [int(self.downsampling_factor * i) for i in self.crop_size])
            resize = TF.resize(crop, self.crop_size, antialias=True)
            for f in range(2 if self.add_flipped_viewport else 1):
                if f==1:
                    resize = TF.hflip(resize)
                four_crop = TF.five_crop(resize, self.target_size)[:-1] # ignor the center crop -1
                for i in range (len(four_crop)):
                    #new_x = copy.copy(x) # non flipped crop
                    #new_x['image'] 
                    result[n] = TF.rotate(four_crop[i], ViewpointTransformation.ROTATIONS[i])
                    n = n+1
                    #result.append(new_x)
        x['image'] = result
        return x