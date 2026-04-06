import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim


class UNetWrapper(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module = nn.BCEWithLogitsLoss(),
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, x):
        return self.model(x)

    def reconstruct(self, x):
        return self.forward(x)

    def _common_step(self, batch, batch_idx):
        x, y = batch, batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch, batch_idx)

        # Calculate a metric (like IoU or Dice)
        preds = (torch.sigmoid(logits) > 0.5).float()
        iou = self._calculate_iou(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_iou", iou, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

    def _calculate_iou(self, preds, target):
        intersection = (preds * target).sum()
        union = preds.sum() + target.sum() - intersection
        return (intersection + 1e-6) / (union + 1e-6)
