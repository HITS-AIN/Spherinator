# Spherinator

Provides simple autoencoders to project images to the surface of a sphere inluding a tool to creat HiPS representations for browsing.

![HiPSter example](efigi.png "Example of autoencoded HiPS tiling for efigi data of nearby galaxies in SDSS")


## Conda environment

Based on [Miniconda](https://docs.conda.io/en/latest/miniconda.html) all dependencies can be installed in a conda environment. The environment can be created and activated with the following commands:

```bash
conda env create
conda activate spherinator
```

## LightningCLI

[LightningCLI](https://lightning.ai/docs/pytorch/latest/cli/lightning_cli.html#lightning-cli) is a command line interface separating source code from hyperparameters. Hyperparameters are defined in a YAML file `config.yaml` and passed to the CLI.

```bash
python main.py fit -c experiments/efigi.yaml
```

## Git submodules

This repository contains git submodules. To clone the repository including the submodules use the following command:

```bash
git clone --recurse-submodules https://github.com/HITS-AIN/Spherinator.git
```

or after cloning with

```bash
git submodule init
git submodule update
```
