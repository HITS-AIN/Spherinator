from models import RotationalAutoencoder
import torch

def test_forward():

    model = RotationalAutoencoder()
    input = model.example_input_array

    recon, coord = model(input)

    assert coord.shape == (1,3)
    assert recon.shape == input.shape

def test_reconstruction_loss():

    torch.manual_seed(0)
    model = RotationalAutoencoder()
    image1 = torch.zeros((2,3,64,64))
    image2 = torch.ones((2,3,64,64))
    image3 = torch.zeros((2,3,64,64))
    image3[0,0,0,0] = 1.0

    assert torch.isclose(model.reconstruction_loss(image1, image1), torch.Tensor([0., 0.]), atol = 1e-3).all()
    assert torch.isclose(model.reconstruction_loss(image1, image2), torch.Tensor([1., 1.]), atol = 1e-3).all()
    assert torch.isclose(model.reconstruction_loss(image1, image3), torch.Tensor([0.009, 0.]), atol = 1e-3).all()
