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
    encoder = ConvolutionalEncoder1D(input_dim=12, output_dim=3)
    decoder = ConvolutionalDecoder1D(input_dim=3, output_dim=12)
    model = VariationalAutoencoderPure(
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


def test_training(parquet_1d_metadata):
    """Test training of VariationalAutoencoderPure"""
    encoder = ConvolutionalEncoder1D(input_dim=12, output_dim=3)
    decoder = ConvolutionalDecoder1D(input_dim=3, output_dim=12)
    model = VariationalAutoencoderPure(
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
        enable_checkpointing=False,
        accelerator="cpu",
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=datamodule)


def test_training_sampling(parquet_test_sampling):
    """Test training of VariationalAutoencoderPure"""
    encoder = ConvolutionalEncoder1D(input_dim=12, output_dim=3)
    decoder = ConvolutionalDecoder1D(input_dim=3, output_dim=12)
    model = VariationalAutoencoderPure(
        encoder=encoder,
        decoder=decoder,
        encoder_out_dim=3,
        loss="KL",
    )

    datamodule = ParquetDataModule(
        parquet_test_sampling,
        data_column="flux",
        error_column="flux_error",
        batch_size=2,
        num_workers=1,
    )

    trainer = Trainer(
        max_epochs=1,
        enable_model_summary=False,
        enable_checkpointing=False,
        accelerator="cpu",
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=datamodule)


def test_training_fixed_scale(parquet_test_sampling):
    """Test training of VariationalAutoencoderPure"""
    encoder = ConvolutionalEncoder1D(input_dim=12, output_dim=3)
    decoder = ConvolutionalDecoder1D(input_dim=3, output_dim=12)
    model = VariationalAutoencoderPure(
        encoder=encoder,
        decoder=decoder,
        encoder_out_dim=3,
        z_dim=3,
        loss="KL",
        fixed_scale=1e3,
    )

    datamodule = ParquetDataModule(
        parquet_test_sampling,
        data_column="flux",
        error_column="flux_error",
        batch_size=2,
        num_workers=1,
    )

    trainer = Trainer(
        max_epochs=1,
        enable_model_summary=False,
        enable_checkpointing=False,
        accelerator="cpu",
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=datamodule)
