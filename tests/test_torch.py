import torch

def test_MSELoss():

    image1 = torch.Tensor([0.0])
    image2 = torch.Tensor([0.1])

    loss = torch.nn.MSELoss(reduction='none')

    assert loss(image1, image1).mean() == 0.0
    assert torch.isclose(loss(image1, image2).mean(), torch.Tensor([0.01]), rtol = 1e-3)

def test_isclose():

    assert torch.isclose(torch.Tensor([1.00001]), torch.Tensor([1.0]), rtol = 1e-5) == False
    assert torch.isclose(torch.Tensor([1.00001]), torch.Tensor([1.0]), rtol = 1e-4)

    assert torch.allclose(torch.Tensor([1.0, 1.00001]), torch.Tensor([1.0, 1.0]), rtol = 1e-5) == False
    assert torch.allclose(torch.Tensor([1.0, 1.0]), torch.Tensor([1.0, 1.0]), rtol = 1e-5)

    assert torch.allclose(torch.tensor([10000., 1e-07]), torch.tensor([10000.1, 1e-08])) == False
    assert torch.allclose(torch.tensor([10000., 1e-08]), torch.tensor([10000.1, 1e-09]))
