import gc

import matplotlib
import torch
import torchvision.transforms.functional as functional
from lightning.pytorch.callbacks import Callback
from matplotlib import figure
from torchvision import transforms

matplotlib.use("Agg")


class LogReconstructionCallback(Callback):
    def __init__(self, num_samples=4):
        super().__init__()
        self.num_samples = num_samples

    def on_train_epoch_end(self, trainer, pl_module):
        # Return if no wandb logger is used
        if trainer.logger is None or trainer.logger.__class__.__name__ not in [
            "WandbLogger",
            "MyLogger",
        ]:
            return

        # Generate some random samples from the validation set
        data, _ = next(iter(trainer.train_dataloader))
        samples = data[: self.num_samples].to(pl_module.device)

        # Generate reconstructions of the samples using the model
        with torch.no_grad():
            best_recon_loss = torch.ones(samples.shape[0], device=samples.device) * 1e10
            best_scaled = torch.zeros(
                samples.shape[0],
                samples.shape[1],
                pl_module.get_input_size(),
                pl_module.get_input_size(),
                device=samples.device,
            )
            best_recon = best_scaled.clone()

            for r in range(pl_module.rotations):
                rotate = functional.rotate(
                    samples, 360.0 / pl_module.rotations * r, expand=False
                )
                crop = functional.center_crop(
                    rotate, [pl_module.crop_size, pl_module.crop_size]
                )
                scaled = functional.resize(
                    crop, [pl_module.input_size, pl_module.input_size], antialias=True
                )

                if pl_module.__class__.__name__ == "RotationalAutoencoder":
                    recon, _ = pl_module(scaled)
                else:
                    (_, _), (_, _), _, recon = pl_module(scaled)

                loss_recon = pl_module.reconstruction_loss(scaled, recon)
                best_recon_idx = torch.where(loss_recon < best_recon_loss)
                best_recon_loss[best_recon_idx] = loss_recon[best_recon_idx]
                best_scaled[best_recon_idx] = scaled[best_recon_idx]
                best_recon[best_recon_idx] = recon[best_recon_idx]

            normalize = transforms.Lambda(
                lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x))
            )
            normalize(best_scaled)
            normalize(best_recon)

        # Plot the original samples and their reconstructions side by side
        fig = figure.Figure(figsize=(6, 2 * self.num_samples))
        ax = fig.subplots(self.num_samples, 2)
        for i in range(self.num_samples):
            ax[i, 0].imshow(best_scaled[i].cpu().detach().numpy().T)
            ax[i, 0].set_title("Original")
            ax[i, 0].axis("off")
            ax[i, 1].imshow(best_recon[i].cpu().detach().numpy().T)
            ax[i, 1].set_title("Reconstruction")
            ax[i, 1].axis("off")
        fig.tight_layout()

        # Log the figure at W&B
        trainer.logger.log_image(key="Reconstructions", images=[fig])

        # Clear the figure and free memory
        # Memory leak issue: https://github.com/matplotlib/matplotlib/issues/27138
        for i in range(self.num_samples):
            ax[i, 0].clear()
            ax[i, 1].clear()
        fig.clear()
        gc.collect()
