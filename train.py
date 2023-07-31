import io
import numpy
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

from models import RotationalSphericalProjectingAutoencoder

import DataSets
import Preprocessing


class GalaxyZooDataModule(pl.LightningDataModule):
    def train_dataloader(self):
        self.dataset = DataSets.GalaxyZooDataset(
            data_directory="/hits/basement/ain/Data/efigi-1.6/png",
                #KaggleGalaxyZoo/images_training_rev1",
                #efigi-1.6/png",
                #KaggleGalaxyZoo/images_training_rev1",
            extension=".png",
            #label_file="/hits/basement/ain/Data/KaggleGalaxyZoo/training_solutions_rev1.csv",
            transform = transforms.Compose([
                Preprocessing.DielemanTransformation(rotation_range=[0,360],
                    translation_range=[4./424,4./424], scaling_range=[1/1.1,1.1], flip=0.5),
                #Preprocessing.KrizhevskyColorTransformation(
                    # weights=[-0.0148366, -0.01253134, -0.01040762], std=0.5),
                Preprocessing.CropAndScale((424,424), (424,424))
            ])
        )

        dataloader = DataLoader(self.dataset,
                                batch_size=64,
                                shuffle=True,
                                num_workers=16)
        return dataloader

if __name__ == "__main__":
    torch.manual_seed(2345)# 2341 2344
    data = GalaxyZooDataModule()
    model = RotationalSphericalProjectingAutoencoder()
    #checkpoint = torch.load("epoch=2611-step=182840.ckpt")
    #model.load_state_dict(checkpoint["state_dict"])
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=-1) #accelerator="cpu" devices=1
    trainer.fit(model, data)
