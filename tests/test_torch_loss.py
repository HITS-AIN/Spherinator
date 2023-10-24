import torch
import torch.nn as nn

def test_MSELoss():

    image1 = torch.Tensor([0.0])
    image2 = torch.Tensor([0.1])

    loss = nn.MSELoss(reduction='none')

    assert loss(image1, image1).mean() == 0.0
    assert torch.isclose(loss(image1, image2).mean(), torch.Tensor([0.01]), rtol = 1e-3)
