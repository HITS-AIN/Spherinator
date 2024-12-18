import torch
from lightning.pytorch.trainer import Trainer

from spherinator.data import ParquetDataModule
from spherinator.models import (
    AutoencoderPure,
    ConvolutionalDecoder1D,
    ConvolutionalEncoder1D,
)


def test_forward():
    """Test forward method of AutoencoderPure"""
    encoder = ConvolutionalEncoder1D(input_dim=12, output_dim=3)
    decoder = ConvolutionalDecoder1D(input_dim=3, output_dim=12)
    model = AutoencoderPure(encoder=encoder, decoder=decoder)
    input = torch.randn(2, 1, 12)

    recon = model(input)
    assert recon.shape == input.shape


def test_training(parquet_1d_metadata):
    """Test training of AutoencoderPure"""
    encoder = ConvolutionalEncoder1D(input_dim=12, output_dim=3)
    decoder = ConvolutionalDecoder1D(input_dim=3, output_dim=12)
    model = AutoencoderPure(encoder=encoder, decoder=decoder)

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
