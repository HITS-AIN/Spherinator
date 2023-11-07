from callbacks import LogReconstructionCallback
from models import RotationalVariationalAutoencoderPower
from data import ShapesDataModule
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import CSVLogger

def test_on_train_epoch_end():

    # Set up the model and dataloader
    z_dim = 3
    model = RotationalVariationalAutoencoderPower(z_dim=z_dim)

    datamodule = ShapesDataModule("/hits/basement/its/doserbd/projects/data/machine-learning/pink/shapes/",
                                   num_workers=1, batch_size=12)
    datamodule.setup("fit")
    # data_loader = data_module.train_dataloader()

    logger = CSVLogger("logs", name="my_exp_name")

    trainer = Trainer(max_epochs=1, logger=logger, overfit_batches = 2)
    trainer.fit(model, datamodule=datamodule)

    # Set up the callback
    num_samples = 2
    callback = LogReconstructionCallback(num_samples=num_samples)

    # Call the callback
    callback.on_train_epoch_end(trainer=trainer, pl_module=model)

    logger.finalize("success")

    # # Check that the figure was logged
    # # assert len(logger.) == 1
    # # assert "Reconstructions" in logger.logged_items[0][0]
    # # assert isinstance(logger.logged_items[0][1], plt.Figure)