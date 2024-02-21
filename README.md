[![Build Status](https://github.com/HITS-AIN/Spherinator/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/HITS-AIN/Spherinator/actions/workflows/python-package.yml?branch=main)
![versions](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)

# Spherinator & HiPSter

`Spherinator` uses [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) to implement a convolutional neural network (CNN) based variational autoencoder (VAE) with a spherical latent space.
`HiPSter` creates the HiPS tilings and the catalog which can be visualized interactively on the surface of a sphere with [Aladin Lite](https://github.com/cds-astro/aladin-lite).

<p align="center">
  <img src="docs/P404_f2.png" width="400" height="400">
</p>


## Installation

[Poetry](https://python-poetry.org/) is used for installation.

```bash
poetry install
```


## LightningCLI

[LightningCLI](https://lightning.ai/docs/pytorch/latest/cli/lightning_cli.html#lightning-cli) is a command line interface separating source code from hyperparameters. Hyperparameters are defined in a YAML file `config.yaml` and passed to the CLI.

```bash
python spherinator.py fit -c experiments/illustris_power.yaml
```

Arguments can be directly defined on the command line and overwrite the YAML file.
Examples:

- Define number of latent dimensions: `--model.init_args.z_dim 16`
- Define GPU indices: `--trainer.devices [0,1]`
- Define number of epochs: `--trainer.max_epochs 100`


## Generate HiPS and catalog

The following command generates a HiPS representation and a catalog showing the real images located on the latent space using the trained model.

```bash
./hipster.py --checkpoint <checkpoint-file>.ckpt
```

Call `./hipster.py --help` for more information.

For visualization, a Python HTTP server can be started by executing `python3 -m http.server 8082` within the HiPSter output file.


## Profiling

The Pytorch profiler can be used by appending the `profiler_pytorch.yaml` config file to the command line.

```bash
python spherinator.sh fit -c experiments/shapes_power.yaml -c experiments/profiler_pytorch.yaml
```


## Visualize reconstructed images during training

The config-file [callback_log_reconstructions.yaml](experiments/callback_log_reconstructions.yaml) can be appended to visualize the reconstructed images during training at W&B. Therefore, the W&B config-file must be appended as well.

```bash
python spherinator.sh fit -c experiments/illustris_power.yaml \
    -c experiments/wandb.yaml \
    -c experiments/callback_log_reconstructions.yaml
```

## Acknowledgements

Funded by the European Union. This work has received funding from the European High Performance Computing Joint Undertaking (JU) and Belgium, Czech Republic, France, Germany, Greece, Italy, Norway, and Spain under grant agreement No 101093441.

Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European High Performance Computing Joint Undertaking (JU) and Belgium, Czech Republic, France, Germany, Greece, Italy, Norway, and Spain. Neither the European Union nor the granting authority can be held responsible for them. 

## License

This project is licensed under the [Apache-2.0 License](http://www.apache.org/licenses/LICENSE-2.0).
