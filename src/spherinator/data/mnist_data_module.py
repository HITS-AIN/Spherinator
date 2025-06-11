from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data/",
        random_rotation: bool = True,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        transformations = [
            transforms.ToTensor(),
            transforms.Pad((0, 0, 1, 1), fill=0),
            transforms.Resize((87, 87)),
        ]
        if random_rotation:
            transformations += [transforms.RandomAffine(degrees=[0, 360])]
        transformations += [
            transforms.Resize((29, 29)),
            transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),  # Normalize to [0, 1]
        ]
        self.transform = transforms.Compose(transformations)

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.mnist_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
