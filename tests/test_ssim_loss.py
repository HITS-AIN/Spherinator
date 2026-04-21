import torch

from spherinator.losses import SSIMLoss


class TestSSIMLoss:
    def test_identical_inputs_returns_zero(self):
        loss = SSIMLoss()
        x = torch.rand(2, 1, 32, 32)
        assert torch.isclose(loss(x, x), torch.tensor(0.0), atol=1e-5)

    def test_different_inputs_positive(self):
        loss = SSIMLoss()
        x = torch.rand(2, 1, 32, 32)
        y = torch.rand(2, 1, 32, 32)
        result = loss(x, y)
        assert result > 0

    def test_output_is_scalar(self):
        loss = SSIMLoss()
        x = torch.rand(2, 1, 32, 32)
        y = torch.rand(2, 1, 32, 32)
        assert loss(x, y).shape == torch.Size([])

    def test_multichannel(self):
        loss = SSIMLoss()
        x = torch.rand(2, 3, 32, 32)
        y = torch.rand(2, 3, 32, 32)
        result = loss(x, y)
        assert result.shape == torch.Size([])
        assert result > 0

    def test_gradient_flows(self):
        loss = SSIMLoss()
        x = torch.rand(2, 1, 32, 32, requires_grad=True)
        y = torch.rand(2, 1, 32, 32)
        loss(x, y).backward()
        assert x.grad is not None

    def test_custom_window_size(self):
        loss = SSIMLoss(window_size=7, sigma=1.0)
        x = torch.rand(2, 1, 32, 32)
        y = torch.rand(2, 1, 32, 32)
        assert loss(x, y).shape == torch.Size([])
