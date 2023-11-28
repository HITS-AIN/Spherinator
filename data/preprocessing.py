import numpy
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class DielemanTransformation:
    def __init__(self, rotation_range, translation_range, scaling_range, flip):
        self.scaling_range = scaling_range
        self.random_affine = transforms.RandomAffine(
            degrees=rotation_range, translate=translation_range, shear=None
        )
        self.flip = transforms.RandomHorizontalFlip(p=flip)

    def __call__(self, images):
        transformed_image = self.random_affine(images)
        zoom = numpy.exp(
            numpy.random.uniform(
                numpy.log(self.scaling_range[0]), numpy.log(self.scaling_range[1])
            )
        )
        resize = TF.resize(
            transformed_image,
            [int(images.shape[1] * zoom), int(images.shape[2] * zoom)],
            antialias=True,
        )
        images = self.flip(resize)
        return images


class KrizhevskyColorTransformation:
    """
    Performs a random brightness scaling based on the specied weights and standard deviation
    following the PCA based idea by Alex Krizhevsky et al. 2012 as used by Dieleman et al. 2015.
    """

    def __init__(self, weights, std):
        self.weights = torch.tensor(numpy.array(weights))
        self.std = std

    def __call__(self, images):
        noise = torch.normal(0.0, self.std, size=[1]) * self.weights
        images[0] = images[0] + noise[0]
        images[1] = images[1] + noise[1]
        images[2] = images[2] + noise[2]
        images = torch.clip(images, 0, 1)
        return images


class CreateNormalizedColors:
    def __init__(self, stretch, range, lower_limit, channel_combinations, scalers):
        """
        Initialize CreateNormalizedColors.

        Args:
            stretch (bool): Flag indicating whether to stretch the image.
            range (tuple): Range of pixel values to be used for stretching.
            lower_limit (int): Lower limit for pixel values.
            channel_combinations (list): List of channel combinations to be used.
            scalers (list): List of scalers to be applied.
        """
        self.stretch = stretch
        self.range = range
        self.lower_limit = lower_limit
        self.channel_combinations = channel_combinations
        self.scalers = scalers

    def __call__(self, images):
        resulting_image = torch.zeros(
            (
                len(self.channel_combinations),
                images.shape[1],
                images.shape[2],
            )
        )
        for i, channel_combination in enumerate(self.channel_combinations):
            resulting_image[i] = images[channel_combination[0]]
            for t in range(1, len(channel_combination)):
                resulting_image[i] = resulting_image[i] + images[channel_combination[t]]
            resulting_image[i] = resulting_image[i] * self.scalers[i]

        mean = torch.mean(resulting_image, dim=0)
        resulting_image = (
            resulting_image
            * torch.asinh(self.stretch * self.range * (mean - self.lower_limit))
            / self.range
            / mean
        )

        resulting_image = torch.nan_to_num(resulting_image, nan=0, posinf=0, neginf=0)
        resulting_image = torch.clip(resulting_image, 0, 1)
        return resulting_image


class ViewpointTransformation:
    ROTATIONS = [0, 90, 270, 180]

    def __init__(
        self,
        target_size,
        crop_size,
        downsampling_factor,
        rotation_angles=[0],
        add_flipped_viewport=False,
    ):
        self.target_size = target_size
        self.crop_size = crop_size
        self.downsampling_factor = downsampling_factor
        self.rotation_angles = rotation_angles
        self.add_flipped_viewport = add_flipped_viewport

    def __call__(self, images):
        result = torch.zeros(
            4 * len(self.rotation_angles) * (2 if self.add_flipped_viewport else 1),
            3,
            self.target_size[0],
            self.target_size[1],
        )
        n = 0
        for angle in self.rotation_angles:
            rotation = TF.rotate(images, angle)
            crop = TF.center_crop(
                rotation, [int(self.downsampling_factor * i) for i in self.crop_size]
            )
            resize = TF.resize(crop, self.crop_size, antialias=True)
            for f in range(2 if self.add_flipped_viewport else 1):
                if f == 1:
                    resize = TF.hflip(resize)
                four_crop = TF.five_crop(resize, self.target_size)[
                    :-1
                ]  # ignor the center crop -1
                for i in range(len(four_crop)):
                    # new_x = copy.copy(x) # non flipped crop
                    # new_x['image']
                    result[n] = TF.rotate(
                        four_crop[i], ViewpointTransformation.ROTATIONS[i]
                    )
                    n = n + 1
                    # result.append(new_x)
        return result
