# Spherinator: Model Training

Spherinator provides representation learning using autoencoders to compress generic data to a
low-dimensional latent space. The primary model is a Variational Autoencoder (VAE) with a
**spherical latent space** based on a
[Power Spherical](https://github.com/nicola-decao/power_spherical) distribution, which is
particularly well-suited for the interactive visualization of high-dimensional data such as images,
spectra, point clouds, and cubes.

```{figure} assets/vae.svg
---
name: fig:vae
align: center
---
```

The encoder compresses the input into a location vector on the unit hypersphere and a concentration
scale. The decoder reconstructs the input from a sample drawn from that distribution. The
`VariationalEncoder` wraps any backbone encoder with two linear heads:

- **`fc_location`** — maps the backbone output to a unit-normalized location vector of dimension
  `z_dim`.
- **`fc_scale`** — maps the backbone output to a positive concentration scalar (via `softplus + 1`
  to avoid collapse).

## Installation

Spherinator can be installed via `pip`:

```bash
pip install spherinator
```

## Training the model

Training requires a YAML configuration file that specifies the data, model architecture, and
training parameters. Multiple config files can be composed on the command line; later files
override earlier ones.

```bash
spherinator fit -c config.yaml
```

Individual arguments can be overridden inline:

```bash
spherinator fit -c config.yaml \
  --model.init_args.z_dim 16 \
  --trainer.devices [0,1] \
  --trainer.max_epochs 100
```

Configs can be **composed** by chaining multiple `-c` flags:

```bash
spherinator fit \
  -c experiments/illustris.yaml \
  -c experiments/vae_vit.yaml \
  -c experiments/wandb.yaml
```

Training can be resumed from a checkpoint:

```bash
spherinator fit -c config.yaml --ckpt_path path/to/checkpoint.ckpt
```

## DataModule

The `DataModule` loads data from [Apache Parquet](https://parquet.apache.org/) files, applies
optional per-column transforms, and feeds batches to the model. It is defined in the `data` section
of the YAML file.

```yaml
data:
  class_path: spherinator.data.DataModule
  init_args:
    path: ./data/illustris_SKIRT_synthetic_images/parquet-128
    columns:
      - name: data
        transform:
          class_path: torchvision.transforms.v2.Resize
          init_args:
            size: 224
    return_dict: False
    batch_size: 16
    shuffle: True
    num_workers: 4
```

| Parameter | Description |
|-----------|-------------|
| `path` | Path to the directory containing Parquet files |
| `columns` | List of columns to load; each entry may carry a `transform` |
| `return_dict` | If `True`, batches are dicts keyed by column name; if `False`, the first column tensor is returned directly |
| `batch_size` | Number of samples per mini-batch |
| `shuffle` | Shuffle the dataset each epoch |
| `num_workers` | Number of parallel data-loading workers |

## Model architecture

The top-level model is defined in the `model` section of the YAML file.
Two model classes are available:

| Class | Description |
|-------|-------------|
| `spherinator.models.Autoencoder` | Deterministic autoencoder |
| `spherinator.models.VariationalAutoencoder` | VAE with Power Spherical latent distribution |


### Autoencoder

The deterministic autoencoder encodes directly to a fixed-size vector without sampling:

```yaml
model:
  class_path: spherinator.models.Autoencoder
  init_args:
    encoder:
      class_path: ...
    decoder:
      class_path: ...
```


### VariationalAutoencoder

```yaml
model:
  class_path: spherinator.models.VariationalAutoencoder
  init_args:
    encoder:
      class_path: ...   # any encoder below
    decoder:
      class_path: ...   # any decoder below
    encoder_out_dim: 64 # must match encoder output_dim
    z_dim: 3            # latent space dimensionality
    beta: 1.0e-3        # KL weight (beta-VAE)
    loss: MSE           # MSE | NLL-normal | NLL-truncated | KL
    fixed_scale: null   # fix concentration (null = learnable)
```

| Parameter | Description |
|-----------|-------------|
| `encoder_out_dim` | Must match the `output_dim` of the encoder backbone |
| `z_dim` | Dimension of the spherical latent space; 3 maps to a sphere (S²) |
| `beta` | Scales the KL divergence term relative to the reconstruction loss |
| `loss` | Reconstruction loss: `MSE`, `NLL-normal`, `NLL-truncated`, or `KL` |
| `fixed_scale` | If set to a float, the concentration is frozen at that value |


## Encoder architectures

The encoder architecture should be chosen to match the input data type.

### ConvolutionalEncoder2D

Standard 2D CNN encoder built from `ConsecutiveConv2DLayer` blocks. Each block is a sequence of
`LazyConv2d` layers with optional batch normalization, activation, and pooling. The output is
flattened and projected to `output_dim` via a lazy linear layer.

```yaml
encoder:
  class_path: spherinator.models.ConvolutionalEncoder2D
  init_args:
    input_dim: [3, 128, 128]
    output_dim: 64
    cnn_layers:
      - class_path: spherinator.models.ConsecutiveConv2DLayer
        init_args:
          kernel_size: 3
          stride: 1
          padding: 0
          out_channels: [16, 20, 24]
      - class_path: spherinator.models.ConsecutiveConv2DLayer
        init_args:
          kernel_size: 4
          stride: 2
          padding: 0
          out_channels: [32, 64]
```

Each entry in `out_channels` adds one convolutional layer. The `ConsecutiveConv2DLayer` arguments
are:

| Argument | Description |
|----------|-------------|
| `kernel_size` | Kernel size for all layers in this block |
| `stride` | Stride for all layers in this block |
| `padding` | Padding for all layers in this block |
| `out_channels` | List of output channel counts; one layer per entry |
| `activation` | Activation function class (default: `nn.ReLU`) |
| `norm` | Normalization class (default: `nn.BatchNorm2d`) |
| `pooling` | Optional pooling module appended after each layer |


### ConvolutionalEncoder1D

Analogous to `ConvolutionalEncoder2D` but for 1D inputs such as spectra, using
`ConsecutiveConv1DLayer` with `LazyConv1d` layers.


### HuggingFaceViTEncoder

Wraps any HuggingFace Vision Transformer. The CLS token from the last hidden state is optionally
projected to `output_dim` via a linear layer.

```yaml
encoder:
  class_path: spherinator.models.HuggingFaceViTEncoder
  init_args:
    model_name: google/vit-base-patch16-224
    output_dim: 64
    freeze: False
```

| Argument | Description |
|----------|-------------|
| `model_name` | HuggingFace model identifier; must be a ViT variant |
| `output_dim` | Output projection size; if `null`, uses the model's hidden size |
| `freeze` | If `True`, the ViT backbone weights are frozen |


## Decoder architectures

### ConvolutionalDecoder2D

Transposed-convolution decoder. A linear layer re-shapes the latent vector to the seed spatial
tensor; `ConsecutiveConvTranspose2DLayer` blocks then upsample to the target resolution.

```yaml
decoder:
  class_path: spherinator.models.ConvolutionalDecoder2D
  init_args:
    input_dim: 3
    output_dim: [3, 128, 128]
    cnn_input_dim: [64, 28, 28]
    cnn_layers:
      - class_path: spherinator.models.ConsecutiveConvTranspose2DLayer
        init_args:
          kernel_size: 5
          stride: 2
          padding: 0
          out_channels: [32]
      - class_path: spherinator.models.ConsecutiveConvTranspose2DLayer
        init_args:
          kernel_size: 3
          stride: 1
          padding: 0
          out_channels: [20, 16, 3]
          activation: null
```

`cnn_input_dim` sets the shape `[C, H, W]` that the seed linear projection reshapes to. The overall
spatial path must reach `output_dim[1:]` through the stacked transpose-conv blocks.

### UpsamplingDecoder2D

Bilinear-upsampling decoder that avoids the checkerboard artifacts of transposed convolutions. Each
`_UpsampleBlock` doubles the spatial resolution with a bilinear upsample followed by a 3×3
convolution, batch normalization, and ReLU. A final 1×1 convolution maps to the output channels,
followed by a sigmoid.

```
z → Linear → reshape (base_channels, seed_size, seed_size)
  → n × UpsampleBlock (2×)
  → 1×1 Conv → Sigmoid → output
```

The number of upsampling steps is inferred automatically to reach `output_dim[1]` from `seed_size`.

```yaml
decoder:
  class_path: spherinator.models.UpsamplingDecoder2D
  init_args:
    input_dim: 64
    output_dim: [3, 224, 224]
    base_channels: 512
    seed_size: 7
```

| Argument | Description |
|----------|-------------|
| `input_dim` | Latent vector size |
| `output_dim` | Target image shape `[C, H, W]` |
| `base_channels` | Channel count at the spatial seed; halved at each upsampling step |
| `seed_size` | Spatial width/height of the seed feature map |

The default `seed_size: 7` with five upsampling steps reaches 224×224
($7 \times 2^5 = 224$), which matches the ViT-Base patch grid.


### ConvolutionalDecoder1D

Mirror of `ConvolutionalDecoder2D` for 1D outputs using `ConsecutiveConvTranspose1DLayer`.


## Optimizer

Any PyTorch optimizer can be specified in the `optimizer` section:

```yaml
optimizer:
  class_path: torch.optim.Adam
  init_args:
    lr: 0.001
```

## Trainer

The `trainer` section maps directly to
[Lightning's Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html):

```yaml
trainer:
  max_epochs: 500
  accelerator: gpu
  devices: auto
  precision: bf16-mixed
  enable_progress_bar: True
  enable_model_summary: True
```

Common precision options: `32`, `16-mixed`, `bf16-mixed`.

## Weights & Biases integration

Append `experiments/wandb.yaml` to enable [W&B](https://wandb.ai) logging. Edit `entity` and
`tags` as needed:

```yaml
# experiments/wandb.yaml
trainer:
  logger:
    class_path: lightning.pytorch.loggers.WandbLogger
    init_args:
      project: spherinator
      log_model: True
      entity: <your-wandb-entity>
      tags:
        - my_experiment
```


## Callbacks

Callbacks are appended as additional YAML files or inline under `trainer.callbacks`.

### Log reconstructions during training

`experiments/callback_log_reconstructions.yaml` logs a fixed set of sample reconstructions to W&B
after every validation epoch. Requires W&B to be configured.

```yaml
# experiments/callback_log_reconstructions.yaml
trainer:
  callbacks:
    - class_path: spherinator.callbacks.LogReconstructionCallback
      init_args:
        samples: 6
```

### Save the best model checkpoint

`experiments/callback_best_model.yaml` saves the checkpoint with the lowest `train_loss`:

```yaml
# experiments/callback_best_model.yaml
trainer:
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        monitor: train_loss
        filename: "{epoch}-{train_loss:.2f}"
        save_top_k: 1
        mode: min
        every_n_epochs: 1
```

## Complete example configurations

### VAE with CNN encoder (128×128 images)

```bash
spherinator fit \
  -c experiments/vae_cnn3.yaml \
  -c experiments/illustris_small.yaml \
  -c experiments/wandb.yaml \
  -c experiments/callback_best_model.yaml
```

### VAE with Vision Transformer encoder (224×224 images)

```bash
spherinator fit \
  -c experiments/vae_vit.yaml \
  -c experiments/illustris_small.yaml \
  -c experiments/wandb.yaml \
  -c experiments/callback_log_reconstructions.yaml
```
