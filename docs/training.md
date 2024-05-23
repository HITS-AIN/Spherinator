# Spherinator Model Training

## LightningCLI

[LightningCLI](https://lightning.ai/docs/pytorch/latest/cli/lightning_cli.html#lightning-cli) is a command line interface separating source code from hyperparameters. Hyperparameters are defined in a YAML file `config.yaml` and passed to the CLI.

```bash
spherinator fit -c experiments/illustris_power.yaml
```

Arguments can be directly defined on the command line and overwrite the YAML file.
Examples:

- Define number of latent dimensions: `--model.init_args.z_dim 16`
- Define GPU indices: `--trainer.devices [0,1]`
- Define number of epochs: `--trainer.max_epochs 100`


## Visualize reconstructed images during training

The config-file [callback_log_reconstructions.yaml](experiments/callback_log_reconstructions.yaml) can be appended to visualize the reconstructed images during training at W&B. Therefore, the W&B config-file must be appended as well.

```bash
spherinator fit -c experiments/illustris_power.yaml \
    -c experiments/wandb.yaml \
    -c experiments/callback_log_reconstructions.yaml
```
