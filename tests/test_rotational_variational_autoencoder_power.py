from models import RotationalVariationalAutoencoderPower
import torch


def test_forward():
    z_dim = 2
    model = RotationalVariationalAutoencoderPower(z_dim=z_dim)
    input = model.example_input_array
    batch_size = input.shape[0]

    (z_mean, z_var), (_, _), _, recon = model(input)

    assert z_mean.shape == (batch_size, z_dim)
    assert z_var.shape == (batch_size, 1)
    assert recon.shape == input.shape


def test_reconstruction_loss():
    z_dim = 2
    model = RotationalVariationalAutoencoderPower(z_dim=z_dim)
    image1 = torch.zeros((2, 3, 128, 128))
    image2 = torch.ones((2, 3, 128, 128))
    image3 = torch.zeros((2, 3, 128, 128))
    image3[0, 0, 0, 0] = 1.0

    assert torch.isclose(
        model.reconstruction_loss(image1, image1), torch.Tensor([0.0, 0.0]), atol=1e-3
    ).all()
    assert torch.isclose(
        model.reconstruction_loss(image1, image2), torch.Tensor([1.0, 1.0]), atol=1e-3
    ).all()
    assert torch.isclose(
        model.reconstruction_loss(image1, image3), torch.Tensor([0.009, 0.0]), atol=1e-2
    ).all()
