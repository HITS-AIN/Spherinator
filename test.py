import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms
from torchvision.datasets import MNIST

import data.MNISTDataModule as MNISTDataModule
import models

# hidden dimension and dimension of latent space
H_DIM = 128
Z_DIM = 5

model = models.SVAE(h_dim=H_DIM, z_dim=Z_DIM, distribution='normal')
# model = models.SVAE(h_dim=H_DIM, z_dim=Z_DIM, distribution='vmf')
# model = models.VAE(latent_dim=1024, input_height=32, input_width=32, input_channels=1, lr=0.0001, batch_size=32)

# input_sample = torch.randn(1, 28, 28)
# _, _, _, output = model(input_sample)
# print(output.shape)
# quit()

# filepath = "svae.onnx"
# model.to_onnx(filepath, input_sample, export_params=True, verbose=True)

# data = MNISTDataModule.MNISTDataModule()
# data.setup('fit')
# print(len(data.train_dataloader()))
# data.train_dataloader()

dataset = MNIST('./data', download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset, num_workers=12)

trainer = pl.Trainer(accelerator='gpu', devices=1)
trainer.fit(model, train_dataloaders=train_loader)
