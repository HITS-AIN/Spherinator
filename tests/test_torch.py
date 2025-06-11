import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler


def test_MSELoss():
    image1 = torch.Tensor([0.0])
    image2 = torch.Tensor([0.1])

    loss = torch.nn.MSELoss(reduction="none")

    assert loss(image1, image1).mean() == 0.0
    assert torch.isclose(loss(image1, image2).mean(), torch.Tensor([0.01]), rtol=1e-3)


def test_isclose():
    # fmt: off
    assert not torch.isclose(torch.Tensor([1.00001]),
                             torch.Tensor([1.0]), rtol=1e-5)
    assert torch.isclose(torch.Tensor([1.00001]),
                         torch.Tensor([1.0]), rtol=1e-4)

    assert not torch.allclose(torch.Tensor([1.0, 1.00001]),
                              torch.Tensor([1.0, 1.0]), rtol=1e-5)
    assert torch.allclose(torch.Tensor([1.0, 1.0]),
                          torch.Tensor([1.0, 1.0]), rtol=1e-5)

    assert not torch.allclose(torch.tensor([10000.0, 1e-07]),
                              torch.tensor([10000.1, 1e-08]))
    assert torch.allclose(torch.tensor([10000.0, 1e-08]),
                          torch.tensor([10000.1, 1e-09]))
    # fmt: on


class MyDataset(Dataset):
    def __init__(self):
        self.data = torch.arange(100).view(100, 1).float()
        self.calls = 0
        self.current_index = []

    def __getitem__(self, index):
        self.current_index = index
        x = self.data[index]
        self.calls += 1
        return x

    def __len__(self):
        return len(self.data)


def test_batch_sampler():
    dataset = MyDataset()
    sampler = BatchSampler(RandomSampler(dataset), batch_size=10, drop_last=False)

    loader = DataLoader(dataset, sampler=sampler)
    batch = next(iter(loader))

    assert batch.shape == (1, 10, 1)
    assert dataset.current_index == [26, 88, 59, 58, 73, 11, 65, 2, 84, 79]
    assert batch.tolist() == [[[26.0], [88.0], [59.0], [58.0], [73.0], [11.0], [65.0], [2.0], [84.0], [79.0]]]
    assert dataset.calls == 1


def test_subtensors_by_index():
    data = torch.Tensor([1, 2, 3, 4, 5])
    index = [2, 3]
    assert torch.allclose(data[index], torch.Tensor([3, 4]))
