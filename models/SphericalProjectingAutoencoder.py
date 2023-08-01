import pytorch_lightning as pl
import torch
import torch.linalg
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SphericalProjectingAutoencoder(pl.LightningModule):

    def __init__(self):
        super(SphericalProjectingAutoencoder, self).__init__()
        self.bottleneck = 3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5,5), stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5,5), stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5,5), stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(5,5), stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(5,5), stride=2, padding=2)
        self.bn5 = nn.BatchNorm2d(1024)
        self.fc1 = nn.Linear(1024*2*2, self.bottleneck)

        self.fc3 = nn.Linear(self.bottleneck, 1024*2*2)
        self.bn6 = nn.BatchNorm2d(1024)
        self.deconv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(5,5), stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(5,5), stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(5,5), stride=2, padding=1)
        self.bn9 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(5,5), stride=2, padding=1)
        self.bn10 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(5,5), stride=2, padding=1)
        self.bn11 = nn.BatchNorm2d(32)
        self.deconv6 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(4,4), stride=1, padding=1)

    def encode(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = x.view(-1,1024*2*2)
        x = F.relu(self.fc1(x))
        return x

    def scale_to_unity(self, x):
        length = torch.linalg.vector_norm(x, dim=1)+1.e-20
        return (x.T / length).T

    def decode(self, x):
        x = self.fc3(x)
#        x = self.fc4(x)
        x = F.relu(self.bn6(x.view(-1, 1024, 2, 2)))
        x = F.relu(self.bn7(self.deconv1(x)))
        x = F.relu(self.bn8(self.deconv2(x)))
        x = F.relu(self.bn9(self.deconv3(x)))
        x = F.relu(self.bn10(self.deconv4(x)))
        x = F.relu(self.bn11(self.deconv5(x)))
        x = self.deconv6(x)
        return x

    def forward(self, x):
        coordinates = self.encode(x)
        return self.decode(self.scale_to_unity(coordinates)), coordinates

    def SphericalLoss(self, input, output, coordinates, rotation=0):
        coord_regularization = torch.square(1 - torch.sum(torch.square(coordinates), dim=1))
        if (rotation != 0):
            output = transforms.functional.rotate(output, rotation, expand=False)
        output = transforms.functional.center_crop(output, (64,64))
        loss = torch.sqrt(torch.sum(torch.square(input.reshape(-1,3*64*64)-output.reshape(-1,3*64*64)), dim=-1)) + coord_regularization
        return loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1.e-3)
        self.scheduler = ReduceLROnPlateau(optimizer=self.optimizer, mode="min", factor=0.1, patience=100, cooldown=20, min_lr=1.e-6, verbose=True)
        return [self.optimizer], [{'scheduler': self.scheduler, 'interval': 'epoch', 'monitor': 'train_loss'}]

    def training_step(self, train_batch, batch_idx):
        images = train_batch['image']#.type(dtype=torch.float32)
        reconstruction, coordinates = self.forward(images)
        rotations = 16
        losses = torch.zeros(images.shape[0],rotations)
        for i in range(rotations): # calculate loss for n rotations and minimize
            losses[:,i] = self.SphericalLoss(images, reconstruction, coordinates, 360.0/rotations*i)
        loss = torch.mean(torch.min(losses, dim=1)[0])
        self.log('train_loss', loss)
        self.log('learning_rate', self.optimizer.state_dict()['param_groups'][0]['lr'])
        return loss
