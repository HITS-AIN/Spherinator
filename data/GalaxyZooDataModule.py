import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchvision import transforms

import data.DataSets as DataSets
import data.Preprocessing as Preprocessing

class GalaxyZooDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = "./", batch_size: int = 32, extension: str = "jpg", shuffle: bool = True, num_workers: int = 16):
        super().__init__()
        self.data_dir = data_dir
        self.train_transform = transforms.Compose([
            Preprocessing.DielemanTransformation( # TODO: in config file?
                rotation_range=[0,360],
                translation_range=[4./424,4./424],
                scaling_range=[1/1.1,1.1],
                flip=0.5),
            Preprocessing.CropAndScale((424,424), (424,424))
        ])
        self.val_transform = transforms.Compose([
            Preprocessing.CropAndScale((424,424), (424,424))
        ])
        self.batch_size = batch_size
        self.extension = extension
        self.shuffle = shuffle
        self.num_workers = num_workers

    def setup(self, stage: str):
        data_full = DataSets.GalaxyZooDataset(data_directory=self.data_dir,
                                              extension=self.extension,
                                              transform=self.train_transform)
        data_raw = DataSets.GalaxyZooDataset(data_directory=self.data_dir,
                                             extension=self.extension,
                                             transform=self.val_transform)
        self.data_train = data_full
        self.data_val = data_raw
        self.data_test = data_full
        self.data_predict = data_full

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.data_predict, batch_size=self.batch_size)
