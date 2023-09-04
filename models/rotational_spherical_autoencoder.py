import lightning.pytorch as pl
import torch
import torch.linalg
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as functional

class RotationalSphericalAutoencoder(pl.LightningModule):

    def __init__(self):
        super(RotationalSphericalAutoencoder, self).__init__()
        self.bottleneck = 3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5,5), stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5), stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5,5), stride=2, padding=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5,5), stride=2, padding=2)
        #self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(5,5), stride=2, padding=2)
        self.fc1 = nn.Linear(256*4*4, 256)
        self.fc2 = nn.Linear(256, self.bottleneck)
        self.fc3 = nn.Linear(self.bottleneck, 256)
        self.fc4 = nn.Linear(256, 256*4*4)
        #self.deconv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(5,5), stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4,4), stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(4,4), stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(4,4), stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(4,4), stride=2, padding=1)
        self.deconv6 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=(5,5), stride=1, padding=2)

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
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

    def spherical_loss(self, images, reconstruction, coordinates):
        coord_regularization = torch.square(1 - torch.sum(torch.square(coordinates), dim=1)) * 1e-4
        loss = torch.sqrt(torch.sum(torch.square(images.reshape(-1,3*64*64)-reconstruction.reshape(-1,3*64*64)), dim=-1)) + coord_regularization
        return loss

    def training_step(self, train_batch, _batch_idx):
        images = train_batch['image']
        rotations = 36
        losses = torch.zeros(images.shape[0], rotations)
        for i in range(rotations):
            rotate = functional.rotate(images, 360/rotations*i, expand=False) # rotate
            crop = functional.center_crop(rotate, [256,256]) # crop
            scaled = functional.resize(crop, [64,64], antialias=False) # scale
            reconstruction, coordinates = self.forward(scaled)
            losses[:,i] =  self.spherical_loss(scaled, reconstruction, coordinates)
        loss = torch.mean(torch.min(losses, dim=1)[0])
<<<<<<< HEAD:models/RotationalSphericalProjectingAutoencoder.py
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        #self.log('learning_rate', self.optimizer.state_dict()['param_groups'][0]['lr'])
=======
        self.log('train_loss', loss)
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'])
>>>>>>> origin/main:models/rotational_spherical_projecting_autoencoder.py
        return loss
