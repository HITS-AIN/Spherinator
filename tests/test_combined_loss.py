import pytest
import torch
import torch.nn as nn

from spherinator.losses import CombinedLoss


class TestCombinedLoss:
    def test_weighted_sum(self):
        loss = CombinedLoss([nn.MSELoss(), nn.L1Loss()], [1.0, 2.0])
        x = torch.ones(4, 3)
        target = torch.zeros(4, 3)
        result = loss(x, target)
        expected = 1.0 * nn.MSELoss()(x, target) + 2.0 * nn.L1Loss()(x, target)
        assert torch.isclose(result, expected)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            CombinedLoss([nn.MSELoss()], [1.0, 2.0])

    def test_single_loss_passthrough(self):
        mse = nn.MSELoss()
        loss = CombinedLoss([mse], [0.5])
        x = torch.randn(4, 3)
        target = torch.randn(4, 3)
        assert torch.isclose(loss(x, target), 0.5 * mse(x, target))
