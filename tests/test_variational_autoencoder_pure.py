import torch
from lightning.pytorch.trainer import Trainer

from spherinator.data import ParquetDataModule
from spherinator.models import (
    ConvolutionalDecoder1D,
    ConvolutionalEncoder1D,
    VariationalAutoencoderPure,
)


def test_forward():
    """Test forward method of VariationalAutoencoderPure"""
    encoder = ConvolutionalEncoder1D(output_dim=3)
    decoder = ConvolutionalDecoder1D(output_dim=12)
    model = VariationalAutoencoderPure(encoder=encoder, decoder=decoder)
    input = torch.randn(2, 1, 12)

    (z_mean, z_var), (_, _), _, recon = model(input)

    batch_size = input.shape[0]
    assert z_mean.shape == (batch_size, 3)
    assert z_var.shape == (batch_size, 1)
    assert recon.shape == input.shape
    assert recon.shape == input.shape


def test_training(parquet_1d_metadata):
    """Test training of VariationalAutoencoderPure"""
    encoder = ConvolutionalEncoder1D(output_dim=3)
    decoder = ConvolutionalDecoder1D(output_dim=12)
    model = VariationalAutoencoderPure(encoder=encoder, decoder=decoder)

    datamodule = ParquetDataModule(
        parquet_1d_metadata,
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
