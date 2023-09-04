import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchvision import transforms

import data.galaxy_zoo_dataset as galaxy_zoo_dataset
import data.preprocessing as preprocessing

class GalaxyZooDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = "./", batch_size: int = 32, extension: str = "jpg", shuffle: bool = True, num_workers: int = 16):
        super().__init__()
        self.data_dir = data_dir
        self.train_transform = transforms.Compose([
            preprocessing.DielemanTransformation(
                rotation_range=[0,360],
                translation_range=[4./424,4./424],
                scaling_range=[1/1.1,1.1],
                flip=0.5),
            preprocessing.CropAndScale((424,424), (424,424))
        ])
        self.val_transform = transforms.Compose([
            preprocessing.CropAndScale((424,424), (424,424))
        ])
        self.batch_size = batch_size
        self.extension = extension
        self.shuffle = shuffle
        self.num_workers = num_workers

    def setup(self, stage: str):
        if stage == "fit":
            self.data_train = galaxy_zoo_dataset.GalaxyZooDataset(data_directory=self.data_dir,
                                                                  extension=self.extension,
                                                                  transform=self.train_transform)

            self.dataloader_train = DataLoader(self.data_train,
                                               batch_size=self.batch_size,
                                               shuffle=self.shuffle,
                                               num_workers=self.num_workers)
        elif stage =="val":
            self.data_val = galaxy_zoo_dataset.GalaxyZooDataset(data_directory=self.data_dir,
                                                                      extension=self.extension,
                                                                      transform=self.val_transform)
            self.dataloader_val = DataLoader(self.data_train,
                                             batch_size=self.batch_size,
                                             shuffle=False,
                                             num_workers=self.num_workers)

    def train_dataloader(self):
        return self.dataloader_train

    def val_dataloader(self):
        return self.dataloader_val
