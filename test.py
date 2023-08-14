import models
import lightning.pytorch as pl
from lightning.pytorch.demos.mnist_datamodule import MNISTDataModule

# model = models.SVAE(h_dim=256, z_dim=32, distribution='vmf')
model = models.VAE(latent_dim=32, input_height=28, input_width=28, input_channels=1, lr=0.0001, batch_size=32)

data = MNISTDataModule()

trainer = pl.Trainer(accelerator='gpu')
trainer.fit(model, datamodule=data)
