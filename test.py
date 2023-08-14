import lightning.pytorch as pl
import torch
from lightning.pytorch.demos.mnist_datamodule import MNISTDataModule

import models

# hidden dimension and dimension of latent space
H_DIM = 128
Z_DIM = 5

model = models.SVAE(h_dim=H_DIM, z_dim=Z_DIM, distribution='normal')
# model = models.SVAE(h_dim=H_DIM, z_dim=Z_DIM, distribution='vmf')
# model = models.VAE(latent_dim=1024, input_height=32, input_width=32, input_channels=1, lr=0.0001, batch_size=32)

input_sample = torch.randn((1, 784))
output = model(input_sample)

# filepath = "svae.onnx"
# model.to_onnx(filepath, input_sample, export_params=True, verbose=True)

data = MNISTDataModule()
# data.setup('fit')
# print(len(data.train_dataloader()))
# data.train_dataloader()

trainer = pl.Trainer(accelerator='gpu', devices=1)
trainer.fit(model, datamodule=data)
