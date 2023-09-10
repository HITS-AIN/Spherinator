import lightning.pytorch as pl
import yaml

import data
import models

if __name__ == "__main__":

    with open("experiments/illustris-svae.yaml", "r", encoding="utf-8") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)

    model = models.RotationalSphericalVariationalAutoencoder(**(config["model"]["init_args"]))

    data_module = data.IllustrisSdssDataModule(data_directories=["TNG50"], num_workers=8, batch_size=32)

    trainer = pl.Trainer(accelerator='gpu', max_epochs=-1)
    trainer.fit(model, data_module)

    print("Done.")
