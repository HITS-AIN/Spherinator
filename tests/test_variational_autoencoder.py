from lightning.pytorch.trainer import Trainer

from spherinator.data import ParquetDataModule
from spherinator.models import (
    ConvolutionalDecoder1D,
    ConvolutionalEncoder1D,
    VariationalAutoencoder,
)


def test_forward():

    z_dim = 3
    encoder = ConvolutionalEncoder1D(input_dim=128, output_dim=256)
    decoder = ConvolutionalDecoder1D(input_dim=256, output_dim=128)
    model = VariationalAutoencoder(encoder=encoder, decoder=decoder, z_dim=z_dim)
    input = model.example_input_array

    (z_mean, z_var), (_, _), _, recon = model(input)

    batch_size = input.shape[0]
    assert z_mean.shape == (batch_size, z_dim)
    assert z_var.shape == (batch_size, 1)
    assert recon.shape == input.shape


def test_training(parquet_numpy_file):

    encoder = ConvolutionalEncoder1D(input_dim=128, output_dim=256)
    decoder = ConvolutionalDecoder1D(input_dim=256, output_dim=128)
    model = VariationalAutoencoder(encoder=encoder, decoder=decoder)

    datamodule = ParquetDataModule(
        parquet_numpy_file,
        data_column="data",
        batch_size=5,
        num_workers=1,
        shuffle=True,
    )

    trainer = Trainer(
        max_epochs=1,
        overfit_batches=2,
        enable_checkpointing=False,
        accelerator="cpu",
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=datamodule)
