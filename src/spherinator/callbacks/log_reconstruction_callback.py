import gc
from typing import Union

import matplotlib
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from matplotlib import figure

matplotlib.use("Agg")


class LogReconstructionCallback(Callback):
    def __init__(
        self,
        samples: Union[int, list[int]] = 6,
    ):
        """
        Callback that logs the original samples and their reconstructions side by side

        Args:
            samples (Union[int, list[int]], optional): The number of samples or a list of indices to log.
                                                       Defaults to 6.
        """
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
        if any(
            sample >= len(trainer.train_dataloader.dataset) for sample in self.samples
        ):
            raise ValueError("The sample indices must be smaller than the dataset size")

        # Get the samples from the dataset
        images = torch.unsqueeze(trainer.train_dataloader.dataset[self.samples[0]], 0)
        for sample in self.samples[1:]:
            images = torch.cat(
                (images, torch.unsqueeze(trainer.train_dataloader.dataset[sample], 0))
            )

        # Move the samples to the device used by the model
        images = images.to(model.device)

        # Generate reconstructions of the samples using the model
        recon = model.pure_forward(images)
        loss = torch.nn.MSELoss(reduction="none")(images, recon).flatten(1).mean(1)

        # Plot the original samples and their reconstructions side by side
        nb_samples = len(self.samples)
        fig = figure.Figure(figsize=(2 * nb_samples, 6))
        ax = fig.subplots(2, nb_samples).flatten()
        for i in range(nb_samples):
            ax[i].imshow(np.clip(images[i].cpu().detach().numpy().T, 0, 1))
            ax[i].set_title(f"Original {self.samples[i]}")
            ax[i].axis("off")
            ax[i + nb_samples].imshow(np.clip(recon[i].cpu().detach().numpy().T, 0, 1))
            ax[i + nb_samples].set_title(f"Recon ({loss[i]:.4f})")
            ax[i + nb_samples].axis("off")
        fig.tight_layout()

        # Log the figure at W&B
        trainer.logger.log_image(key="Reconstructions", images=[fig])

        # Clear the figure and free memory
        # Memory leak issue: https://github.com/matplotlib/matplotlib/issues/27138
        for i in range(2 * nb_samples):
            ax[i].clear()
        fig.clear()
        gc.collect()
