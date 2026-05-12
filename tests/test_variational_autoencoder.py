import pytest
import torch
from lightning.pytorch.trainer import Trainer

from spherinator.data import ParquetDataModule
from spherinator.models import ConvolutionalDecoder1D, ConvolutionalEncoder1D, VariationalAutoencoder


@pytest.fixture
def encoder():
    """Fixture for encoder for 1D data."""
    return ConvolutionalEncoder1D([1, 12], 3)


@pytest.fixture
def decoder():
    """Fixture for decoder for 1D data."""
    return ConvolutionalDecoder1D(3, [1, 12], [1, 12])


def test_forward(encoder, decoder):
    """Test forward method of VariationalAutoencoder"""
    model = VariationalAutoencoder(
        encoder=encoder,
        decoder=decoder,
        encoder_out_dim=3,
    )
    input = torch.randn(2, 1, 12)

    (z_mean, z_var), (_, _), _, recon = model(input)

    batch_size = input.shape[0]
    assert z_mean.shape == (batch_size, 3)
    assert z_var.shape == (batch_size, 1)
    assert recon.shape == input.shape
    assert recon.shape == input.shape


def test_training(parquet_1d_metadata, encoder, decoder):
    """Test training of VariationalAutoencoder"""
    model = VariationalAutoencoder(
        encoder=encoder,
        decoder=decoder,
        encoder_out_dim=3,
    )

    datamodule = ParquetDataModule(
        parquet_1d_metadata,
        data_column="data",
        batch_size=5,
        num_workers=1,
        shuffle=True,
    )

    trainer = Trainer(
        max_epochs=1,
        enable_model_summary=False,
        accelerator="cpu",
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=datamodule)
