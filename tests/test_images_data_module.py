import numpy as np
import torch

from spherinator.data import ImagesDataModule


def test_fit():
    data = ImagesDataModule(
        "tests/data/images",
        image_size=224,
        num_workers=1,
        batch_size=2,
    )
    data.setup("fit")

    assert len(data.data_train) == 2

    dataloader = data.train_dataloader()

    assert dataloader.batch_size == 2
    assert len(dataloader) == 1
    assert dataloader.num_workers == 1

    batch = next(iter(dataloader))

    assert batch.shape == (2, 3, 224, 224)
    assert batch.dtype == torch.float32

    assert np.isclose(batch.min(), 0.0)
    assert np.isclose(batch.max(), 1.0)
