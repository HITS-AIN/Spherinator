from abc import abstractmethod

import lightning.pytorch as pl


class SpherinatorModule(pl.LightningModule):
    """
    Abstract base class for all spherinator modules.
    """

    @abstractmethod
    def reconstruction_loss(self, images, reconstructions):
        pass

    @abstractmethod
    def project(self, images):
        pass

    @abstractmethod
    def reconstruct(self, coordinates):
        pass
