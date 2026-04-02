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


@pytest.mark.parametrize("loss", ["NLL-normal", "NLL-truncated", "KL"])
def test_training_sampling(parquet_test_sampling, loss, encoder, decoder):
    """Test training of VariationalAutoencoder"""
    model = VariationalAutoencoder(
        encoder=encoder,
        decoder=decoder,
        encoder_out_dim=3,
        loss=loss,
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


def test_training_fixed_scale(parquet_test_sampling, encoder, decoder, tmp_path):
    """Test training of VariationalAutoencoder"""
    model = VariationalAutoencoder(
        encoder=encoder,
        decoder=decoder,
        encoder_out_dim=3,
        z_dim=3,
        loss="KL",
        fixed_scale=1e3,
    )

    assert model.fixed_scale == 1e3
    assert not model.variational_encoder.fc_scale.weight.requires_grad
    assert not model.variational_encoder.fc_scale.bias.requires_grad
    assert torch.all(model.variational_encoder.fc_scale.weight.data == 0)
    assert torch.all(model.variational_encoder.fc_scale.bias.data == 1e3)

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
        default_root_dir=tmp_path,
    )
    trainer.fit(model, datamodule=datamodule)


def test_training_restart_without_fixed_scale(parquet_test_sampling, encoder, decoder, tmp_path):
    """Test training of VariationalAutoencoder with fixed scale and restarting without fixed scale"""
    model = VariationalAutoencoder(
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
        accelerator="cpu",
        log_every_n_steps=1,
        default_root_dir=tmp_path,
    )
    trainer.fit(model, datamodule=datamodule)

    # Restart training without fixed scale
    model_no_fixed_scale = VariationalAutoencoder(
        encoder=encoder,
        decoder=decoder,
        encoder_out_dim=3,
        z_dim=3,
        loss="KL",
    )

    trainer.fit(
        model_no_fixed_scale,
        datamodule=datamodule,
        ckpt_path=tmp_path / "lightning_logs" / "version_0" / "checkpoints" / "epoch=0-step=5.ckpt",
    )

    assert model_no_fixed_scale.fixed_scale == None
    assert model_no_fixed_scale.variational_encoder.fc_scale.weight.requires_grad
    assert model_no_fixed_scale.variational_encoder.fc_scale.bias.requires_grad
    assert torch.all(model.variational_encoder.fc_scale.weight.data == 0)
    assert torch.all(model.variational_encoder.fc_scale.bias.data == 1e3)
