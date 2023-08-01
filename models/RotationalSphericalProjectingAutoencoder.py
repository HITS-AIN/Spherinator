import lightning.pytorch as pl
import torch
import torch.linalg
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau


class RotationalSphericalProjectingAutoencoder(pl.LightningModule):

    def __init__(self):
        super(RotationalSphericalProjectingAutoencoder, self).__init__()
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

    def encode(self, x, rotation):
        x = transforms.functional.rotate(x, rotation, expand=False)
        x = transforms.functional.center_crop(x, (256,256)) # crop
        input = transforms.functional.resize(x,(64,64), antialias=False) #scale

        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        #x = F.relu(self.conv5(x))
        x = x.view(-1,256*4*4)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x, input

    def scale_to_unity(self, x):
        length = torch.linalg.vector_norm(x, dim=1)+1.e-20
        return (x.T / length).T

    def decode(self, x):
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.fc4(x))
        x = x.view(-1, 256, 4, 4)
        #x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = F.relu(self.deconv5(x))
        x = self.deconv6(x)
        return x

    def forward(self, x, rotation = 0.0):
        print(x.shape)
        print(rotation)
        coordinates, input = self.encode(x, rotation)
        return input, self.decode(self.scale_to_unity(coordinates)), coordinates

    def SphericalLoss(self, input, output, coordinates):#, rotation=0):
        coord_regularization = torch.square(1 - torch.sum(torch.square(coordinates), dim=1)) * 1e-4
       # if (rotation != 0):
       #     output = transforms.functional.rotate(output, rotation, expand=False)
        #output = transforms.functional.center_crop(output, (64,64))
        #rot = transforms.functional.rotate(input, rotation, expand=False)
        #crop = transforms.functional.center_crop(rot, (256,256)) # crop
        #scale = transforms.functional.resize(crop,(64,64), antialias=False) #scale

        loss = torch.sqrt(torch.sum(torch.square(input.reshape(-1,3*64*64)-output.reshape(-1,3*64*64)), dim=-1)) + coord_regularization
        return loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1.e-3)
        self.scheduler = ReduceLROnPlateau(optimizer=self.optimizer, mode="min", factor=0.1, patience=500, cooldown=500, min_lr=1.e-5, verbose=True)
        return [self.optimizer], [{'scheduler': self.scheduler, 'interval': 'epoch', 'monitor': 'train_loss'}]

    def training_step(self, train_batch, batch_idx):
        images = train_batch['image']#.type(dtype=torch.float32)
        rotations = 36
        losses = torch.zeros(images.shape[0], rotations)
        for i in range(rotations):
            input, reconstruction, coordinates = self.forward(images, 360.0/rotations*i)
            losses[:,i] =  self.SphericalLoss(input, reconstruction, coordinates)


        #input, reconstruction, coordinates = self.forward(images, rotation)
        #rotations = 1
        #losses = torch.zeros(images.shape[0], rotations)
        #for i in range(rotations): # calculate loss for n rotations and minimize
        #losses = self.SphericalLoss(images, reconstruction, coordinates, rotations)
        loss = torch.mean(torch.min(losses, dim=1)[0])
        self.log('train_loss', loss)
        self.log('learning_rate', self.optimizer.state_dict()['param_groups'][0]['lr'])
        return loss
