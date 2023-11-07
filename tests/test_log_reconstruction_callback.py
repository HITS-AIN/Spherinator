import matplotlib.pyplot as plt
from lightning.pytorch.loggers import Logger
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.utilities import rank_zero_only

from callbacks import LogReconstructionCallback
from data import ShapesDataModule
from models import RotationalVariationalAutoencoderPower


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


def test_on_train_epoch_end():

    # Set up the model and dataloader
    z_dim = 3
    model = RotationalVariationalAutoencoderPower(z_dim=z_dim)

    datamodule = ShapesDataModule("tests/data/shapes", num_workers=1, batch_size=12)
    datamodule.setup("fit")
    # data_loader = data_module.train_dataloader()

    logger = MyLogger()

    trainer = Trainer(max_epochs=1, logger=logger, overfit_batches = 2)
    trainer.fit(model, datamodule=datamodule)

    # Set up the callback
    num_samples = 2
    callback = LogReconstructionCallback(num_samples=num_samples)

    # Call the callback
    callback.on_train_epoch_end(trainer=trainer, pl_module=model)

    logger.finalize("success")

    # Check that the figure was logged
    assert logger.calls == 1
    assert "Reconstructions" in logger.logged_items[0][0]
    assert isinstance(logger.logged_items[0][1][0], plt.Figure)
