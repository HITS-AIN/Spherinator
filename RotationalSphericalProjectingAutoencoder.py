import io
import numpy
import gc
from matplotlib import pyplot

import torch
import torch.linalg
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

import DataSets
import Preprocessing

class GalaxyZooDataModule(pl.LightningDataModule):
    def train_dataloader(self):
        self.dataset = DataSets.GalaxyZooDataset(data_directory="/hits/basement/ain/Data/KaggleGalaxyZoo/images_training_rev1",#efigi-1.6/png",#KaggleGalaxyZoo/images_training_rev1",
                                                 extension=".jpg",
                                                 #label_file="/hits/basement/ain/Data/KaggleGalaxyZoo/training_solutions_rev1.csv",
                                                 transform = transforms.Compose([Preprocessing.DielemanTransformation(rotation_range=[0,360], translation_range=[4./424,4./424], scaling_range=[1/1.1,1.1], flip=0.5),
                                                                                 #Preprocessing.KrizhevskyColorTransformation(weights=[-0.0148366, -0.01253134, -0.01040762], std=0.5),
                                                                                 Preprocessing.CropAndScale((424,424), (424,424))
                                                                                 ])
        )

        dataloader = DataLoader(self.dataset,
                                batch_size=64,
                                shuffle=True,
                                num_workers=16)
        return dataloader
        
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
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = F.relu(self.deconv5(x))
        x = self.deconv6(x)
        return x
     
    def forward(self, x, rotation):
        coordinates, input = self.encode(x, rotation)
        return input, self.decode(self.scale_to_unity(coordinates)), coordinates
        
    def SphericalLoss(self, input, output, coordinates):#, rotation=0):
        coord_regularization = torch.square(1 - torch.sum(torch.square(coordinates), dim=1)) * 1e-4
        loss = torch.sqrt(torch.sum(torch.square(input.reshape(-1,3*64*64)-output.reshape(-1,3*64*64)), dim=-1)) + coord_regularization
        return loss
        
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1.e-3)
        self.scheduler = ReduceLROnPlateau(optimizer=self.optimizer, mode="min", factor=0.1, patience=500, cooldown=500, min_lr=1.e-5, verbose=True)
        return [self.optimizer], [{'scheduler': self.scheduler, 'interval': 'epoch', 'monitor': 'train_loss'}]
    
    def training_step(self, train_batch, batch_idx):
        images = train_batch['image']
        rotations = 36
        losses = torch.zeros(images.shape[0], rotations)
        for i in range(rotations):
            input, reconstruction, coordinates = self.forward(images, 360.0/rotations*i)
            losses[:,i] =  self.SphericalLoss(input, reconstruction, coordinates)
        loss = torch.mean(torch.min(losses, dim=1)[0])
        self.log('train_loss', loss)
        self.log('learning_rate', self.optimizer.state_dict()['param_groups'][0]['lr'])
        return loss

    def project_dataset(self, dataloader, rotation_steps):
        result_coordinates = torch.zeros((0, 3))
        result_rotations = torch.zeros((0))
        for batch in dataloader:
            print(".", end="")
            losses = torch.zeros((batch['id'].shape[0],rotation_steps))
            coords = torch.zeros((batch['id'].shape[0],rotation_steps,3))
            for r in range(rotation_steps):
                 input, reconstruction, coordinates = self.forward(batch['image'], 360.0/rotation_steps*r)
                 input = input.detach()
                 reconstruction = reconstruction.detach()
                 coordinates = coordinates.detach()
                 losses[:,r] = self.SphericalLoss(input, reconstruction, coordinates)
                 coords[:,r] = self.scale_to_unity(coordinates)
                 del input
                 del reconstruction
                 del coordinates
                 self.zero_grad()
            min = torch.argmin(losses, dim=1)
            result_coordinates = torch.cat((result_coordinates, coords[torch.arange(batch['id'].shape[0]),min]))
            result_rotations = torch.cat((result_rotations, 360.0/rotation_steps*min))
            del losses
            del coords
            del min
            gc.collect()
        return result_coordinates, result_rotations
       
if __name__ == "__main__":
    torch.manual_seed(2341)# 2341 2344 
    data = GalaxyZooDataModule()
    model = RotationalSphericalProjectingAutoencoder()
    checkpoint = torch.load("gz_epoch514-step124115.ckpt")
    model.load_state_dict(checkpoint["state_dict"])
    trainer = pl.Trainer(max_epochs=-1) #accelerator="gpu", devices=1
    trainer.fit(model, data)
        