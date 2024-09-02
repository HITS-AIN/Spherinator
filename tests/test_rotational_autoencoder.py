import torch
from lightning.pytorch.trainer import Trainer

from spherinator.data import ImagesDataModule
from spherinator.models import RotationalAutoencoder


def test_forward():
    model = RotationalAutoencoder()
    input = model.example_input_array
    recon = model(input)
    assert recon.shape == input.shape


def test_training(shape_path):
    model = RotationalAutoencoder(norm_brightness=True)

    datamodule = ImagesDataModule(
        "tests/data/images",
        image_size=224,
        num_workers=1,
        batch_size=2,
    )

    trainer = Trainer(
        max_epochs=1,
        overfit_batches=2,
        enable_checkpointing=False,
        accelerator="cpu",
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=datamodule)


def test_reconstruction_loss():
    model = RotationalAutoencoder()
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
