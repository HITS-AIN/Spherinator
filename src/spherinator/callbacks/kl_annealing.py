import lightning.pytorch as pl


class KLAnnealing(pl.Callback):
    """Callback to anneal the KL divergence weight (beta) during training.

    Args:
        start: Initial value of beta at the start of training.
        end: Final value of beta at the end of training.
        n_epochs: Total number of epochs for training.
        n_cycles: Number of cycles to repeat the annealing schedule over n_epochs.
        ratio: Fraction of each cycle spent ramping from start to end (rest stays at end).
    """

    def __init__(
        self,
        start: float = 0.0,
        end: float = 1.0e-2,
        n_epochs: int = 100,
        n_cycles: int = 1,
        ratio: float = 1.0,
    ):
        self.start = start
        self.end = end
        self.n_epochs = n_epochs
        self.n_cycles = n_cycles
        self.ratio = ratio

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        epoch = trainer.current_epoch
        if epoch >= self.n_epochs:
            pl_module.beta = self.end
        else:
            cycle_length = self.n_epochs / self.n_cycles
            cycle_pos = (epoch % cycle_length) / cycle_length
            t = min(cycle_pos / self.ratio, 1.0)
            pl_module.beta = self.start + t * (self.end - self.start)
        pl_module.log("beta", pl_module.beta)
