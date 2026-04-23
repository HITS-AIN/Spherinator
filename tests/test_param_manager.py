import lightning.pytorch as pl
import pytest
import torch
import torch.nn as nn

from spherinator.callbacks.param_manager import ParamConfig, ParamManager


class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([5.0]))
        self.bias = nn.Parameter(torch.tensor([3.0]))

    def forward(self, x):
        return x * self.weight + self.bias

    def training_step(self, batch, batch_idx):
        return self(batch).sum()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


@pytest.fixture
def model():
    return SimpleModel()


@pytest.fixture
def trainer():
    return pl.Trainer(
        max_epochs=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        accelerator="cpu",
    )


def test_set_value_on_train_start(model, trainer):
    callback = ParamManager(configs=[ParamConfig(pattern="weight", value=2.0)])
    callback.on_train_start(trainer, model)
    assert model.weight.item() == pytest.approx(2.0)


def test_freeze_on_train_start(model, trainer):
    callback = ParamManager(configs=[ParamConfig(pattern="weight", freeze=True)])
    callback.on_train_start(trainer, model)
    assert not model.weight.requires_grad
    assert model.bias.requires_grad  # unmatched param unaffected


def test_max_value_clamps_after_batch(model, trainer):
    # Set weight above max_value, then trigger on_train_batch_end
    with torch.no_grad():
        model.weight.fill_(10.0)

    callback = ParamManager(configs=[ParamConfig(pattern="weight", max_value=3.0)])
    callback.on_train_batch_end(trainer, model, outputs=None, batch=None, batch_idx=0)

    assert model.weight.item() <= 3.0


def test_max_value_does_not_clamp_below_max(model, trainer):
    # Value below max_value should remain unchanged
    with torch.no_grad():
        model.weight.fill_(1.0)

    callback = ParamManager(configs=[ParamConfig(pattern="weight", max_value=3.0)])
    callback.on_train_batch_end(trainer, model, outputs=None, batch=None, batch_idx=0)

    assert model.weight.item() == pytest.approx(1.0)


def test_max_value_none_skips_clamp(model, trainer):
    # No max_value set; on_train_batch_end should leave param untouched
    with torch.no_grad():
        model.weight.fill_(100.0)

    callback = ParamManager(configs=[ParamConfig(pattern="weight")])
    callback.on_train_batch_end(trainer, model, outputs=None, batch=None, batch_idx=0)

    assert model.weight.item() == pytest.approx(100.0)


def test_pattern_does_not_match_other_params(model, trainer):
    with torch.no_grad():
        model.bias.fill_(10.0)

    callback = ParamManager(configs=[ParamConfig(pattern="weight", max_value=3.0)])
    callback.on_train_batch_end(trainer, model, outputs=None, batch=None, batch_idx=0)

    # bias should be untouched
    assert model.bias.item() == pytest.approx(10.0)


def test_set_value_and_max_value_combined(model, trainer):
    callback = ParamManager(configs=[ParamConfig(pattern="weight", value=5.0, max_value=3.0)])
    callback.on_train_start(trainer, model)
    assert model.weight.item() == pytest.approx(3.0)  # value set then clamped to max_value on start

    # after a batch, clamped to max
    callback.on_train_batch_end(trainer, model, outputs=None, batch=None, batch_idx=0)
    assert model.weight.item() == pytest.approx(3.0)
