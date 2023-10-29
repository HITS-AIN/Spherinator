import torch
import torch.linalg
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as functional

from .spherinator_module import SpherinatorModule


class RotationalAutoencoder(SpherinatorModule):

    def __init__(self,
                 rotations: int = 36,
                 bottleneck: int = 3):
        super(RotationalAutoencoder, self).__init__()
        self.bottleneck = bottleneck
        self.rotations = rotations
        self.input_size = 128
        self.example_input_array = torch.randn(1, bottleneck, self.input_size, self.input_size)
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=16,
                               kernel_size=(3,3), stride=1, padding=1) #128x128
        self.pool0 = nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0) # 64x64
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=(3,3), stride=1, padding=1) #64x64
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0) # 32x32
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=(3,3), stride=1, padding=1) #32x32
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0) # 16x16
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(3,3), stride=1, padding=1) #16x16
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0) # 8x8
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=(3,3), stride=1, padding=1) #8x8
        self.pool4 = nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0) # 4x4

        self.fc1 = nn.Linear(256*4*4, 256)
        self.fc2 = nn.Linear(256, self.bottleneck)
        self.fc3 = nn.Linear(self.bottleneck, 256)
        self.fc4 = nn.Linear(256, 256*4*4)

        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128,
                                          kernel_size=(4,4), stride=2, padding=1) #8x8
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=(4,4), stride=2, padding=1) #16x16
        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                          kernel_size=(4,4), stride=2, padding=1) #32x32
        self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=32,
                                          kernel_size=(4,4), stride=2, padding=1) #64x64
        self.deconv5 = nn.ConvTranspose2d(in_channels=32, out_channels=16,
                                          kernel_size=(3,3), stride=2, padding=1) #127x127
        self.deconv6 = nn.ConvTranspose2d(in_channels=16, out_channels=3,
                                          kernel_size=(2,2), stride=1, padding=0) #128x128

    def get_input_size(self):
        return self.input_size

    def encode(self, x):
        x = F.relu(self.conv0(x))
        x = self.pool0(x)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = x.view(-1,256*4*4)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

    def scale_to_unity(self, x):
        length = torch.linalg.vector_norm(x, dim=1)+1.e-20
        return (x.T / length).T

    def decode(self, x):
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.fc4(x))
        x = x.view(-1, 256, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = F.relu(self.deconv5(x))
        x = self.deconv6(x)
        return x

    def forward(self, x):
        coordinates = self.encode(x)
        reconstruction = self.decode(self.scale_to_unity(coordinates))
        return reconstruction, coordinates

    def spherical_loss(self, images, reconstructions, coordinates):
        coord_regularization = torch.square(1 - torch.sum(torch.square(coordinates), dim=1))
        reconstruction_loss = self.reconstruction_loss(images, reconstructions)
        loss = reconstruction_loss + 1e-4 * coord_regularization
        return loss

    def training_step(self, train_batch, _batch_idx):
        images = train_batch['image']
        losses = torch.zeros(images.shape[0], self.rotations)
        for i in range(self.rotations):
            rotate = functional.rotate(images, 360.0/self.rotations*i, expand=False) # rotate
            crop = functional.center_crop(rotate, [256,256]) # crop #TODO config?
            scaled = functional.resize(crop, [self.input_size, self.input_size], antialias=False) # scale
            reconstruction, coordinates = self.forward(scaled)
            losses[:,i] = self.spherical_loss(scaled, reconstruction, coordinates)
        loss = torch.mean(torch.min(losses, dim=1)[0])
        self.log('train_loss', loss)
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'])
        return loss

    def project(self, images):
        return self.scale_to_unity(self.encode(images))

    def reconstruct(self, coordinates):
        return self.decode(coordinates)

    def reconstruction_loss(self, images, reconstructions):
        return torch.sqrt(torch.mean(torch.square(images.reshape(-1,3*images.shape[-2]*images.shape[-1]) -
                                                  reconstructions.reshape(-1,3*images.shape[-2]*images.shape[-1])), dim=-1))
