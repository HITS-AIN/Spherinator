import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms
from torchvision.datasets import MNIST

import data
import models

# hidden dimension and dimension of latent space
H_DIM = 128
Z_DIM = 5

# model = models.VAE(latent_dim=1024, input_height=32, input_width=32, input_channels=1, lr=0.0001, batch_size=32)
model = models.SVAE(h_dim=H_DIM, z_dim=Z_DIM, distribution='normal')
# model = models.SVAE(h_dim=H_DIM, z_dim=Z_DIM, distribution='vmf')

# dataset = MNIST('./data', download=True, transform=transforms.ToTensor())
# train_loader = torch.utils.data.DataLoader(dataset, num_workers=12, batch_size=32)

# trainer = pl.Trainer(accelerator='gpu', devices=1)
# trainer.fit(model, train_dataloaders=train_loader)

data = data.MNISTDataModule(batch_size=32, num_workers=4)

trainer = pl.Trainer(accelerator='gpu', devices=1)
trainer.fit(model, datamodule=data)
