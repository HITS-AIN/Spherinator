from models import RotationalVariationalAutoencoder
import torch

def test_forward():

    z_dim = 2
    model = RotationalVariationalAutoencoder(z_dim=z_dim)
    input = model.example_input_array
    batch_size = input.shape[0]

    (z_mean, z_var), (_, _), _, recon = model(input)

    assert z_mean.shape == (batch_size, z_dim)
    assert z_var.shape == (batch_size, z_dim)
    assert recon.shape == input.shape

def test_reconstruction_loss():

    torch.manual_seed(0)
    z_dim = 2
    model = RotationalVariationalAutoencoder(z_dim=z_dim)
    image1 = torch.zeros((2,3,64,64))
    image2 = torch.ones((2,3,64,64))
    image3 = torch.zeros((2,3,64,64))
    image3[0,0,0,0] = 1.0

    assert model.reconstruction_loss(image1, image1) == 0.0
    assert torch.isclose(model.reconstruction_loss(image1, image2), torch.Tensor([3*64*64]), rtol = 1e-3)
    assert torch.isclose(model.reconstruction_loss(image1, image3), torch.Tensor([0.5]), rtol = 1e-3)
