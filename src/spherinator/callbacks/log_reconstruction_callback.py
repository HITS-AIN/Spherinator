import gc
from typing import Union

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from matplotlib import figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


class LogReconstructionCallback(Callback):
    """
    Callback that logs the original samples and their reconstructions side by side

    Args:
        samples (Union[int, list[int]], optional): The number of samples or a list of indices to log.
                                                    Defaults to 6.
    """

    def __init__(
        self,
        samples: Union[int, list[int]] = 6,
    ):
        super().__init__()

        if isinstance(samples, int):
            if samples < 0:
                raise ValueError("The number of samples must be positive")
            self.samples = range(samples)
        elif isinstance(samples, list):
            if any(sample < 0 for sample in samples):
                raise ValueError("The sample indices must be positive")
            self.samples = samples

    def on_train_epoch_end(self, trainer, model):
        # Return if no wandb logger is used
        if trainer.logger is None or trainer.logger.__class__.__name__ not in [
            "WandbLogger",
            "MyLogger",
        ]:
            return

        # Check sample indices
        if any(sample >= len(trainer.train_dataloader.dataset) for sample in self.samples):
            raise ValueError("The sample indices must be smaller than the dataset size")

        # Get the samples from the dataset
        def _get_pair(item):
            """Return (augmented, original). For plain tensors, both are the same."""
            if isinstance(item, (list, tuple)):
                return item[0], item[-1]
            return item, item

        pairs = [_get_pair(trainer.train_dataloader.dataset[s]) for s in self.samples]
        images_aug = torch.stack([p[0] for p in pairs])
        images_orig = torch.stack([p[1] for p in pairs])

        # Move the samples to the device used by the model
        images_aug = images_aug.to(model.device)
        images_orig = images_orig.to(model.device)

        # Generate reconstructions from augmented images
        recon = model.reconstruct(images_aug)
        loss = torch.nn.MSELoss(reduction="none")(images_orig, recon).flatten(1).mean(1)

        # Plot original, augmented, and reconstruction rows
        nb_samples = len(self.samples)
        fig = figure.Figure(figsize=(2 * nb_samples, 9))
        FigureCanvasAgg(fig)
        ax = fig.subplots(3, nb_samples).flatten()
        for i in range(nb_samples):
            ax[i].imshow(np.clip(images_orig[i].cpu().detach().numpy().transpose(1, 2, 0), 0, 1))
            ax[i].set_title(f"Original {self.samples[i]}")
            ax[i].axis("off")
            ax[i + nb_samples].imshow(np.clip(images_aug[i].cpu().detach().numpy().transpose(1, 2, 0), 0, 1))
            ax[i + nb_samples].set_title(f"Augmented {self.samples[i]}")
            ax[i + nb_samples].axis("off")
            ax[i + 2 * nb_samples].imshow(np.clip(recon[i].cpu().detach().numpy().transpose(1, 2, 0), 0, 1))
            ax[i + 2 * nb_samples].set_title(f"Recon ({loss[i]:.4f})")
            ax[i + 2 * nb_samples].axis("off")
        fig.tight_layout()

        # Log the figure at W&B
        trainer.logger.log_image(key="Reconstructions", images=[fig])

        # Clear the figure and free memory
        # Memory leak issue: https://github.com/matplotlib/matplotlib/issues/27138
        for i in range(3 * nb_samples):
            ax[i].clear()
        fig.clear()
        gc.collect()
