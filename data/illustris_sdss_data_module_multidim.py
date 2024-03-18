from torch.utils.data import DataLoader

from illustris_sdss_data_module import IllustrisSdssDataModule
from illustris_sdss_dataset_multidim import IllustrisSdssDatasetMultidim

class IllustrisSdssDataModuleMultidim(IllustrisSdssDataModule):
    def __init__(
            self,
            data_directories: list[str],
            hp_directory: str,
            info_directory: str,
            extension: str = "fits",
            minsize: int = 100,
            shuffle: bool = True,
            batch_size: int = 32,
            num_workers: int = 16,
    ):
        super().__init__(data_directories, extension, minsize, shuffle, batch_size, num_workers)
        self.hp_directory = hp_directory
        self.info_directory = info_directory

    def setup(self, stage: str):
        if stage == "multidim_processing":
            self.data_multidim = IllustrisSdssDatasetMultidim(
                data_directories=self.data_directories,
                cutout_directory=self.hp_directory,
                info_dir=self.info_directory,
                extension=self.extension,
                minsize=self.minsize,
                transform=self.transform_processing,
            )
            self.dataloader_multidim = DataLoader(
                self.data_multidim,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        else:
            super().setup(stage)

