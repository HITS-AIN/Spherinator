import argparse

import torch
import yaml

import data
import models

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test SVAE")
    parser.add_argument("--config", "-c", default="config.yaml",
        help="config file (default = 'config.yaml').")

    vars = vars(parser.parse_args())
    if "config" in vars:
        with open(vars["config"], "r", encoding="utf-8") as stream:
            config = yaml.load(stream, Loader=yaml.Loader)

    model = models.RotationalVariationalAutoencoder(**(config["model"]["init_args"]))

    checkpoint = torch.load("lightning_logs/version_13/checkpoints/epoch=43-step=3344.ckpt")
    model.load_state_dict(checkpoint["state_dict"])

    data_module = data.IllustrisSdssDataModule(**(config["data"]["init_args"]))
    data_module.setup("predict")
    dataloader = data_module.predict_dataloader()

    for batch in dataloader:
        image = batch["image"]
        z_mean, _ = model.encode(image)
        print(z_mean)
        break
