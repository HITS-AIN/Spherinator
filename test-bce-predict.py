import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as functional

import data
import models

if __name__ == "__main__":

    model = models.RotationalSphericalVariationalAutoencoder.load_from_checkpoint(
        "spherinator/euj7yhpr/checkpoints/epoch=34-step=16205.ckpt")

    data_module = data.IllustrisSdssDataModule(data_directories=["TNG50"], num_workers=1, batch_size=1)
    data_module.setup("fit")
    data_loader = data_module.train_dataloader()

    for batch in data_loader:
        image = batch["image"]
        image = image.to(model.device)
        image = functional.rotate(image, 0.0, expand=False)
        image = functional.center_crop(image, [256,256])
        input = functional.resize(image, [64,64], antialias=False)

        plt.imshow(input.cpu().numpy().squeeze().T)
        plt.show()

        _, (q_z, p_z), _, recon = model(input)

        recon_image = torch.sigmoid(recon)

        plt.imshow(recon_image.cpu().detach().numpy().squeeze().T)
        plt.show()

    print("Done.")
