import pytest
import torch

from spherinator.models import DenseModel


@pytest.mark.parametrize(("layer_dims"), [[2, 3], [3, 2], [56, 28, 3]])
def test_dense_model(layer_dims):
    encoder = DenseModel(layer_dims=layer_dims)
    data = torch.randn([2, layer_dims[0]])
    out = encoder(data)

    assert out.shape == torch.Size([2, layer_dims[-1]])


def test_dense_model_unflatten():
    encoder = DenseModel(layer_dims=[2, 3], output_shape=[1, 3])
    data = torch.randn([2, 2])
    out = encoder(data)

    assert out.shape == torch.Size([2, 1, 3])
