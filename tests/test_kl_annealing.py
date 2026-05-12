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

    def test_cyclic_2(self):
        cb = KLAnnealing(start=0.0, end=1.0e-2, n_epochs=200, n_cycles=4, ratio=1.0)

        epochs = [0, 1, 49, 50, 200]
        expected_betas = [0.0, 0.0002, 0.0098, 0.0, 0.01]

        for epoch, expected_beta in zip(epochs, expected_betas):
            trainer, module = self._make_trainer_and_module(epoch)
            cb.on_train_epoch_start(trainer, module)
            assert module.beta == pytest.approx(expected_beta)
