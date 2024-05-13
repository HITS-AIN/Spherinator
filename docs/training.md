# Training

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
