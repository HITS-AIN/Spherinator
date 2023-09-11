from abc import ABC, abstractmethod

import lightning.pytorch as pl


class SpherinatorModule(ABC, pl.LightningModule):
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
