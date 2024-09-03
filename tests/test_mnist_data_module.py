import numpy as np
import torch

from spherinator.data import MNISTDataModule


def test_fit(tmp_path):
    data = MNISTDataModule(tmp_path, random_rotation=True, num_workers=1, batch_size=2)
    data.prepare_data()
    data.setup("fit")

    assert len(data.mnist_train) == 55000

    dataloader = data.train_dataloader()

    assert dataloader.batch_size == 2
    assert len(dataloader) == 27500
    assert dataloader.num_workers == 1

    batch = next(iter(dataloader))
    images, labels = batch

    assert images.shape == (2, 1, 29, 29)
    assert images.dtype == torch.float32

    assert np.isclose(images.min(), 0.0)
    assert np.isclose(images.max(), 1.0)

    assert (labels == torch.Tensor([1, 3])).all()
