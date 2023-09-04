import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms
from torchvision.datasets import MNIST

import data
import models

data = data.MNISTDataModule(batch_size=32, num_workers=4)
model = models.SVAE.load_from_checkpoint("./lightning_logs/version_3/checkpoints/epoch=340-step=586179.ckpt")

trainer = pl.Trainer(accelerator='gpu', devices=1)
predictions = trainer.predict(model, datamodule=data)
