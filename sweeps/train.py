""" Training function for W&B sweeps
"""

import sys
import importlib
import lightning.pytorch as pl
import torch
import yaml

sys.path.append('../')

torch.set_float32_matmul_precision('high')

def main():
    # Set up your default hyperparameters
    with open("./input.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # cli = LightningCLI(run=False, save_config_kwargs={"overwrite": True}, args=config)
    # cli.trainer.fit(cli.model, cli.datamodule)

    model_class_path = config['model']['class_path']
    module_name, class_name = model_class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    model_init_args = config['model']['init_args']
    model = model_class(**model_init_args)

    data_class_path = config['data']['class_path']
    module_name, class_name = data_class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    data_class = getattr(module, class_name)
    data_init_args = config['data']['init_args']
    data_module = data_class(**data_init_args)
    data_module.setup("fit")

    trainer = pl.Trainer(accelerator="gpu", max_epochs=-1, devices=1)
    trainer.fit(model, datamodule=data_module)

main()
