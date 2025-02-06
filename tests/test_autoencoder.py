import torch
from lightning.pytorch.trainer import Trainer

from spherinator.data import ParquetDataModule
from spherinator.models import (
    Autoencoder,
    ConvolutionalDecoder1D,
    ConvolutionalEncoder1D,
)


def test_forward():
    """Test forward method of Autoencoder"""
    encoder = ConvolutionalEncoder1D(input_dim=12, output_dim=24)
    decoder = ConvolutionalDecoder1D(input_dim=24, output_dim=12)
    model = Autoencoder(encoder=encoder, decoder=decoder, h_dim=24, z_dim=3)
    input = torch.randn(2, 1, 12)

    recon = model(input)
    assert recon.shape == input.shape


def test_training(parquet_1d_metadata):
    """Test training of Autoencoder"""
    encoder = ConvolutionalEncoder1D(input_dim=12, output_dim=24)
    decoder = ConvolutionalDecoder1D(input_dim=24, output_dim=12)
    model = Autoencoder(encoder=encoder, decoder=decoder, h_dim=24, z_dim=3)

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
        enable_checkpointing=False,
        accelerator="cpu",
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=datamodule)
