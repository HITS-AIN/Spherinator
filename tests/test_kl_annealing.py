from unittest.mock import MagicMock

import pytest

from spherinator.callbacks import KLAnnealing


class TestKLAnnealing:
    def _make_trainer_and_module(self, epoch: int):
        trainer = MagicMock()
        trainer.current_epoch = epoch
        module = MagicMock()
        return trainer, module

    def test_beta_at_start(self):
        cb = KLAnnealing(start=0.0, end=1.0, n_epochs=10)
        trainer, module = self._make_trainer_and_module(0)
        cb.on_train_epoch_start(trainer, module)
        assert module.beta == pytest.approx(0.0)

    def test_beta_at_end(self):
        cb = KLAnnealing(start=0.0, end=1.0, n_epochs=10)
        trainer, module = self._make_trainer_and_module(10)
        cb.on_train_epoch_start(trainer, module)
        assert module.beta == pytest.approx(1.0)

    def test_beta_clamped_beyond_n_epochs(self):
        cb = KLAnnealing(start=0.0, end=1.0, n_epochs=10)
        trainer, module = self._make_trainer_and_module(999)
        cb.on_train_epoch_start(trainer, module)
        assert module.beta == pytest.approx(1.0)

    def test_beta_midpoint(self):
        cb = KLAnnealing(start=0.0, end=1.0, n_epochs=10)
        trainer, module = self._make_trainer_and_module(5)
        cb.on_train_epoch_start(trainer, module)
        assert module.beta == pytest.approx(0.5)
