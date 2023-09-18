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


## Conda environment

Based on [Miniconda](https://docs.conda.io/en/latest/miniconda.html) all dependencies can be installed in a conda environment. The environment can be created and activated with the following commands:

```bash
conda env create
conda activate spherinator
```


## LightningCLI

[LightningCLI](https://lightning.ai/docs/pytorch/latest/cli/lightning_cli.html#lightning-cli) is a command line interface separating source code from hyperparameters. Hyperparameters are defined in a YAML file `config.yaml` and passed to the CLI.

```bash
python main.py fit -c experiments/Illustris.yaml
```


## Generate HiPS and catalog

The following command generates a HiPS representation and a catalog showing the real images located on the latent space using the trained model.

```bash
./hipster.py all --checkpoint <checkpoint-file>.ckpt
```

Call `./hipster.py --help` for more information.
