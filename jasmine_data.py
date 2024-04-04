from data import IllustrisSdssDataModuleMultidim
from models import RotationalVariationalAutoencoderPower
from pathlib import Path
from hipster import HipsterMultidim
import yaml
import torch

if __name__ == "__main__":
    config_file = "experiments/tng_multidim_power.yaml"
    with open(config_file, "r", encoding="utf-8") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)
        model_init_args = config["model"]["init_args"]

    model = RotationalVariationalAutoencoderPower(**model_init_args)
    checkpoint = torch.load("tng-models/model.ckpt")
    model.load_state_dict(checkpoint["state_dict"])

    data_init_args = config["data"]["init_args"]
    datamodule = IllustrisSdssDataModuleMultidim(**data_init_args)

    hipster = HipsterMultidim("jasmine-example", "TNG100-h2", verbose=True, max_order=2, hierarchy=2,
                              output_size=256, crop_size=256)
    #hipster.generate_hips(model)
    #hipster.generate_catalog(model, datamodule)
    #hipster.generate_dataset_projection(datamodule)
    #hipster.transform_csv_to_votable()

    hipster.create_data_cube(datamodule, data_aspects=[
        #"morphology",
        ##"gas_pointclouds",
        #"gas_temperature_fields",
        #"dm_pointclouds",
        "stars"
    ])