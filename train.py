import lightning.pytorch as pl
import torch
import yaml

import GalaxyZooDataModule
import models

if __name__ == "__main__":

    torch.manual_seed(2345)

    with open('config.yaml', encoding="utf-8") as file:
        config = yaml.load(file, yaml.SafeLoader)

    # data = GalaxyZooDataModule.GalaxyZooDataModule(
    #     data_dir = "/hits/basement/ain/Data/efigi-1.6/png")
    data = GalaxyZooDataModule.GalaxyZooDataModule(
        data_dir = "/hits/basement/ain/Data/KaggleGalaxyZoo/images_training_rev1", extension="jpg")

    model = models.RotationalSphericalProjectingAutoencoder()

    trainer = pl.Trainer(**config['training'])
    trainer.fit(model, datamodule=data)

    trainer.test(dataloaders=data)
