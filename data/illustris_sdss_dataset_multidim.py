from .illustris_sdss_dataset import IllustrisSdssDataset
from . import visualization as vis
import os
import h5py
from strenum import StrEnum
import csv
import numpy as np
import ast


class TngParticleTypes(StrEnum):
    GAS = "PartType0",
    DM = "PartType1",
    STARS = "PartType4",
    BLACKHOLE = "PartType5"


class IllustrisSdssDatasetMultidim(IllustrisSdssDataset):
    def __init__(
            self,
            data_directories: list[str],
            cutout_directory: str,
            info_dir: str,
            extension: str = "fits",
            minsize: int = 100,
            transform=None,
    ):
        # Superclass constructor call will find FITS files (relevant for training) and store them
        super().__init__(data_directories, extension, minsize, transform)

        # We extend the constructor to store The h5 files with additional info
        self.cutout_directory = cutout_directory
        self.info_dir = info_dir
        self.info = []
        with open(info_dir, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            self.info = [ row for row in reader]

    def get_cutout(self, index: int):
        subhalo_id = self.get_metadata(index)["subhalo_id"]
        h5_path = os.path.join(self.cutout_directory, "cutout_{0}.hdf5".format(subhalo_id))
        cutout = h5py.File(h5_path, 'r')
        return cutout

    def get_metadata(self, index: int):
        filename = self.files[index]
        info_i = self.info[index]
        metadata = {
            "filename": filename,
            "id": index,
            "simulation": info_i["SimulationName"],
            "snapshot": info_i["SnapshotNumber"],
            "subhalo_id": info_i["InfoID"]

        }
        return metadata

    def get_sciencedata(self, index: int):
        info_i = self.info[index]
        science_data = {
            "mass": float(info_i["SubhaloMass"]),
            "velocity": np.array(info_i["SubhaloVel"]),
            "spin": np.array(info_i["SubhaloSpin"]),
            "center_position": np.array(info_i["SubhaloPos"])
        }
        return science_data

    def make_pointcloud(self, index):
        dist_units_kpc = 1.476232654266312
        mass_units_msun = 14762326542.663124

        science_data = self.get_sciencedata(index)
        cutout = self.get_cutout(index)
        if cutout.keys().__contains__(TngParticleTypes.GAS):
            gas = cutout[TngParticleTypes.GAS]
            center_pos = ast.literal_eval(str(science_data["center_position"]))
            gas_coords = (np.array(gas['Coordinates'], dtype=np.float64) - center_pos) * dist_units_kpc
            g_pot = np.array(gas['Potential'], dtype=np.float64)
            g_pot /= g_pot.mean()

            return vis.gas_potential_pointcloud(gas_coords, g_pot)





