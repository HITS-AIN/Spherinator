import lightning.pytorch as pl


class KLAnnealing(pl.Callback):
    def __init__(self, start: float = 0.0, end: float = 1.0e-2, n_epochs: int = 100):
        self.start = start
        self.end = end
        self.n_epochs = n_epochs

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        epoch = trainer.current_epoch
        t = min(epoch / self.n_epochs, 1.0)
        pl_module.beta = self.start + t * (self.end - self.start)
        pl_module.log("beta", pl_module.beta)
