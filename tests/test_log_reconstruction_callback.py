import matplotlib.pyplot as plt
import pytest
from lightning.pytorch.loggers import Logger
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.utilities import rank_zero_only

from spherinator.callbacks import LogReconstructionCallback
from spherinator.data import ParquetDataModule
from spherinator.models import (
    Autoencoder,
    ConvolutionalDecoder1D,
    ConvolutionalEncoder1D,
)


class MyLogger(Logger):
    @property
    def name(self):
        return "MyLogger"

    @property
    def version(self):
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        pass

    @rank_zero_only
    def save(self):
        pass

    @rank_zero_only
    def finalize(self, status):
        pass

    def __init__(self):
        self.calls = 0
        self.logged_items = []

    def log_image(self, key, images):
        self.calls += 1
        self.logged_items.append((key, images))


class MyNullContext:
    def __enter__(self, *args, **kwargs):
        pass

    def __exit__(self, *args, **kwargs):
        pass


does_not_raise = MyNullContext()


@pytest.mark.parametrize(
    "samples, exception",
    [
        (2, does_not_raise),
        (
            5,
            pytest.raises(
                ValueError,
                match=r"The sample indices must be smaller than the dataset size",
            ),
        ),
        ([0, 1], does_not_raise),
        (
            [0, 42],
            pytest.raises(
                ValueError,
                match=r"The sample indices must be smaller than the dataset size",
            ),
        ),
    ],
)
def test_on_train_epoch_end(samples, exception, parquet_2d_metadata):
    # Set up the model and dataloader
    encoder = ConvolutionalEncoder1D(input_dim=12, output_dim=3)
    decoder = ConvolutionalDecoder1D(input_dim=3, output_dim=12)
    model = Autoencoder(encoder=encoder, decoder=decoder)

    datamodule = ParquetDataModule(
        parquet_2d_metadata, data_column="data", batch_size=2
    )

    logger = MyLogger()

    trainer = Trainer(
        max_epochs=1,
        logger=logger,
        overfit_batches=2,
        log_every_n_steps=1,
        enable_checkpointing=False,
        accelerator="cpu",
    )
    trainer.fit(model, datamodule=datamodule)

    # Set up the callback
    callback = LogReconstructionCallback(samples=samples)

    # Call the callback
    with exception:
        callback.on_train_epoch_end(trainer=trainer, model=model)

    if exception != does_not_raise:
        return

    logger.finalize("success")

    # Check that the figure was logged
    assert logger.calls == 1
    assert "Reconstructions" in logger.logged_items[0][0]
    assert isinstance(logger.logged_items[0][1][0], plt.Figure)
