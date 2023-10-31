import torch
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback

class LogReconstructionCallback(Callback):
    def __init__(self, num_samples=4):
        super().__init__()
        self.num_samples = num_samples

    def on_train_epoch_end(self, trainer, pl_module):
        # Generate some random samples from the validation set
        samples = next(iter(pl_module.train_dataloader()))['image']
        samples = samples[:self.num_samples]

        # Generate reconstructions of the samples using the model
        with torch.no_grad():
            (_, _), (_ ,_), _, recon_samples = pl_module(samples)

        # Plot the original samples and their reconstructions side by side
        fig, axs = plt.subplots(self.num_samples, 2, figsize=(6, 2*self.num_samples))
        for i in range(self.num_samples):
            axs[i, 0].imshow(samples[i].permute(1, 2, 0))
            axs[i, 0].set_title("Original")
            axs[i, 0].axis("off")
            axs[i, 1].imshow(recon_samples[i].permute(1, 2, 0))
            axs[i, 1].set_title("Reconstruction")
            axs[i, 1].axis("off")
        plt.tight_layout()

        # Log the figure at W&B
        pl_module.logger.log_image(key='Reconstructions', images=fig)
