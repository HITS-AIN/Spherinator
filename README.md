[![Build Status](https://github.com/HITS-AIN/Spherinator/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/HITS-AIN/Spherinator/actions/workflows/python-package.yml?branch=main)
![versions](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)

# Spherinator

Provides simple autoencoders to project images to the surface of a sphere inluding a tool to creat HiPS representations for browsing.

![HiPSter example](efigi.png "Example of autoencoded HiPS tiling for efigi data of nearby galaxies in SDSS")


## Git clone with submodules

This repository contains git submodules. To clone the repository including the submodules use the following command:

```bash
git clone --recurse-submodules https://github.com/HITS-AIN/Spherinator.git
```

or after cloning with

```bash
git submodule init
git submodule update
```


## Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
... DO YOUR WORK ...
deactivate
```


## Conda environment

Based on [Miniconda](https://docs.conda.io/en/latest/miniconda.html) all dependencies can be installed in a conda environment. The environment can be created and activated with the following commands:

```bash
conda env create
conda activate spherinator
```


## LightningCLI

[LightningCLI](https://lightning.ai/docs/pytorch/latest/cli/lightning_cli.html#lightning-cli) is a command line interface separating source code from hyperparameters. Hyperparameters are defined in a YAML file `config.yaml` and passed to the CLI.

```bash
python spherinator.py fit -c experiments/illustris-power.yaml
```

Arguments can be overwritten on the command line. For example, `--model.init_args.z_dim 16` overwrites the `z_dim` argument in the YAML file.


## Generate HiPS and catalog

The following command generates a HiPS representation and a catalog showing the real images located on the latent space using the trained model.

```bash
./hipster.py all --checkpoint <checkpoint-file>.ckpt
```

Call `./hipster.py --help` for more information.


## Profiling

The Pytorch profiler can be used by appending the `pytorch-profiler.yaml` config file to the command line.

```bash
python main.py fit -c experiments/shapes-power.yaml -c experiments/pytorch-profiler.yaml
```


## Visualize reconstructed images during training

The config-file [wandb-log-reconstructions.yaml](experiments/wandb-log-reconstructions.yaml) can be appended to visualize the reconstructed images during training at W&B.

```bash
python main.py fit -c experiments/illustris.yaml -c experiments/wandb-log-reconstructions.yaml
```
