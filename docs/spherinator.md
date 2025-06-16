# Spherinator: Model Training

Spherinator provide representation learning using a Variational Autoencoder (VAE) to reduce generic
data to a spherical latent space. The spherical latent space is suitable for the visualization of
high-dimensional data, such as images, point clouds, and cubes.

```{figure} assets/vae.svg
---
name: fig:pest
width: 500px
align: center
---
```

## Installation

Sperinator can be installed via `pip`:

```bash
pip install spherinator
```

## LightningCLI

[LightningCLI](https://lightning.ai/docs/pytorch/latest/cli/lightning_cli.html#lightning-cli) is a command line interface separating source code from hyperparameters. The hyperparameters are defined in a YAML file `config.yaml` and passed to the CLI.

```bash
spherinator fit -c experiments/config.yaml
```

Arguments can be directly defined on the command line and overwrite the YAML file.
Examples:

- Define number of latent dimensions: `--model.init_args.z_dim 16`
- Define GPU indices: `--trainer.devices [0,1]`
- Define number of epochs: `--trainer.max_epochs 100`

### Defining a model architecture

The model architecture is defined in the `model` section of the YAML file. The model classes are defined in `spherinator/models/`.

#### Example of a variational autoencoder (VAE) model architecture:
```yaml
model:
  class_path: spherinator.models.VariationalAutoencoder
  init_args:
    encoder:
      ... see section below ...
    decoder:
      ... see section below ...
    z_dim: 3
    beta: 1.0e-4
    encoder_out_dim: 128
```

### Defining an encoder and decoder

The encoder and decoder are defined in the `init_args` section of the model.

#### Example of a 2-dimensional convolutional encoder with 4 layers:
```yaml
encoder:
  class_path: spherinator.models.ConvolutionalEncoder2DGen
  init_args:
    input_dim: [1, 512, 512]
    output_dim: 128
    cnn_layers:
      - class_path: spherinator.models.ConsecutiveConv2DLayer
        init_args:
          kernel_size: 5
          stride: 2
          padding: 0
          num_layers: 4
          base_channel_number: 32
          channel_multiplier: 2
          activation: nn.ReLU
          norm: nn.BatchNorm2d
          pooling: nn.MaxPool2d
```

The `ConsecutiveConv2DLayer` class is a wrapper for multiple convolutional layers. The `num_layers`
argument defines the number of consecutive layers. The `base_channel_number` argument defines the
number of channels in the first layer. The `channel_multiplier` argument defines the
multiplicative factor for the number of channels in the next layer.

## Visualize reconstructed images during training

The config-file [callback_log_reconstructions.yaml](experiments/callback_log_reconstructions.yaml) can be appended to visualize the reconstructed images during training at W&B. Therefore, the W&B config-file must be appended as well.

```bash
spherinator fit -c experiments/config.yaml \
    -c experiments/wandb.yaml \
    -c experiments/callback_log_reconstructions.yaml
```
