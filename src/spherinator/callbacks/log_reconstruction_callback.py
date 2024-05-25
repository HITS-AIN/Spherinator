import gc

import matplotlib
import numpy as np
import torch
import torchvision.transforms.functional as functional
from lightning.pytorch.callbacks import Callback
from matplotlib import figure

matplotlib.use("Agg")


class LogReconstructionCallback(Callback):
    def __init__(
        self,
        num_samples: int = 4,
        indices: list[int] = [],
    ):
        super().__init__()
        self.num_samples = num_samples
        self.indices = indices

    def on_train_epoch_end(self, trainer, model):
        # Return if no wandb logger is used
        if trainer.logger is None or trainer.logger.__class__.__name__ not in [
            "WandbLogger",
            "MyLogger",
        ]:
            return

        # Generate some random samples from the validation set
        data = next(iter(trainer.train_dataloader))
        samples = data[: self.num_samples].to(model.device)

        # Generate reconstructions of the samples using the model
        with torch.no_grad():
            best_recon_loss = torch.ones(samples.shape[0], device=samples.device) * 1e10
            best_scaled = torch.zeros(
                samples.shape[0],
                samples.shape[1],
                model.get_input_size(),
                model.get_input_size(),
                device=samples.device,
            )
            best_recon = best_scaled.clone()

            for r in range(model.rotations):
                rotate = functional.rotate(
                    samples, 360.0 / model.rotations * r, expand=False
                )
                crop = functional.center_crop(
                    rotate, [model.crop_size, model.crop_size]
                )
                scaled = functional.resize(
                    crop, [model.input_size, model.input_size], antialias=True
                )

                z = model.project(scaled)
                recon = model.reconstruct(z)

                loss_recon = model.reconstruction_loss(scaled, recon)
                best_recon_idx = torch.where(loss_recon < best_recon_loss)
                best_recon_loss[best_recon_idx] = loss_recon[best_recon_idx]
                best_scaled[best_recon_idx] = scaled[best_recon_idx]
                best_recon[best_recon_idx] = recon[best_recon_idx]

        # Plot the original samples and their reconstructions side by side
        fig = figure.Figure(figsize=(2 * self.num_samples, 6))
        ax = fig.subplots(2, self.num_samples).flatten()
        for i in range(self.num_samples):
            ax[i].imshow(np.clip(best_scaled[i].cpu().detach().numpy().T, 0, 1))
            ax[i].set_title("Original")
            ax[i].axis("off")
            ax[i + self.num_samples].imshow(
                np.clip(best_recon[i].cpu().detach().numpy().T, 0, 1)
            )
            ax[i + self.num_samples].set_title("Reconstruction")
            ax[i + self.num_samples].axis("off")
        fig.tight_layout()

        # Log the figure at W&B
        trainer.logger.log_image(key="Reconstructions", images=[fig])

        # Clear the figure and free memory
        # Memory leak issue: https://github.com/matplotlib/matplotlib/issues/27138
        for i in range(2 * self.num_samples):
            ax[i].clear()
        fig.clear()
        gc.collect()
