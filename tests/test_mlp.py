import pytest
import torch

from spherinator.models import MLP


@pytest.mark.parametrize(("hidden_sizes"), [[], [2], [3, 2]])
def test_mlp(hidden_sizes):
    input_size = 2
    output_size = 3
    net = MLP(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)
    data = torch.randn([2, input_size])
    out = net(data)

    assert out.shape == torch.Size([2, output_size])


def test_mlp_unflatten():
    net = MLP(input_size=2, hidden_sizes=[3], output_size=3)
    data = torch.randn([2, 2])
    out = net(data)

    assert out.shape == torch.Size([2, 3])
