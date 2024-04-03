from .illustris_sdss_dataset_with_metadata import IllustrisSdssDatasetWithMetadata
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


def gas_temperature(gas):
    # calculate the temperature because apparently, physicians love to do the math by themselves...
    u = np.array(gas["InternalEnergy"])
    x_e = np.array(gas["ElectronAbundance"])
    x_h = 0.76
    gamma = 5/3
    k_b = 1.380649 * 10**(-16)
    mu = (4/(1 + 3*x_h + 4*x_h*x_e)) * np.array(gas["Masses"])
    unit_length = 3.086*(10**21) # 1 kpc
    unit_time = 3.1536*(10**16) # 1 Gyr
    print(unit_length**2 / unit_time**2)
    temp = (gamma -1) * u/k_b * (unit_length**2 / unit_time**2) * mu
    return temp


class IllustrisSdssDatasetMultidim(IllustrisSdssDatasetWithMetadata):
    def __init__(
            self,
            data_directories: list[str],
            cutout_directory: str,
            info_dir: str,
            data_aspect: str = None,
            extension: str = "fits",
            minsize: int = 100,
            transform=None,
    ):
        # Superclass constructor call will find FITS files (relevant for training) and store them
        super().__init__(data_directories, extension, minsize, transform)

        # We extend the constructor to store The h5 files with additional info
        self.cutout_directory = cutout_directory
        self.info_dir = info_dir
        self.data_aspect = data_aspect
        self.info = []
        with open(info_dir, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            self.info = [ row for row in reader]

        self.dist_units_kpc = 1.476232654266312
        self.to_msun = 14762326542.663124

    def __getitem__(self, index: int):
        """Retrieves the item/items with the given indices from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            data: Data of the item/items with the given indices.
            metadata: Metadata of the item/items with the given indices.
        """
        data, metadata = super().__getitem__(index)
        return data, metadata

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

    def get_visual_data(self, index):
        if self.data_aspect is not None:
            science_data = self.get_sciencedata(index)
            cutout = self.get_cutout(index)
            get_data_aspect = getattr(self, f"make_{self.data_aspect}")
            return get_data_aspect(science_data, cutout)

    def make_gas_pointcloud(self, science_data, cutout):
        if cutout.keys().__contains__(TngParticleTypes.GAS):
            gas = cutout[TngParticleTypes.GAS]
            gas_coords = self.center_coordinates(gas['Coordinates'], science_data)
            g_pot = np.array(gas['Potential'], dtype=np.float64)
            g_pot /= g_pot.mean()

            return vis.make_pointcloud(gas_coords, g_pot)

    def make_dm_pointcloud(self, science_data, cutout):
        if cutout.keys().__contains__(TngParticleTypes.DM):
            dm = cutout[TngParticleTypes.DM]
            dm_coords = self.center_coordinates(dm['Coordinates'], science_data)
            d_pot = np.array(dm['Potential'], dtype=np.float64)
            d_pot /= d_pot.mean()

            return vis.make_pointcloud(dm_coords, d_pot)

    def make_star_pointcloud(self, science_data, cutout):
        if cutout.keys().__contains__(TngParticleTypes.STARS):
            stars = cutout[TngParticleTypes.STARS]
            star_coords = self.center_coordinates(stars['Coordinates'], science_data)
            density = np.array(stars['SubfindDensity'])
            mass = np.array(stars['Masses'])
            volume = mass / density
            radius = ((3 / 4) * (volume / np.pi))**(1 / 3) * self.dist_units_kpc

            return vis.star_point_cloud(star_coords, radius)

    def make_gas_temperature_field(self, science_data, cutout):
        if cutout.keys().__contains__(TngParticleTypes.GAS):
            gas = cutout[TngParticleTypes.GAS]
            gas_coords = self.center_coordinates(gas['Coordinates'], science_data)
            temperature = gas_temperature(gas)
            min_max = [-20, 20]

            t = vis.binned_stats_img(gas_coords, np.log(temperature), min_max)
            extent = [min_max[0], min_max[1], min_max[0], min_max[1]]
            return t, extent

    def make_dark_matter_field(self, science_data, cutout):
        if cutout.keys().__contains__(TngParticleTypes.DM):
            dm = cutout[TngParticleTypes.DM]
            dm_coords = self.center_coordinates(dm['Coordinates'], science_data)
            dm_density = np.array(dm['SubfindDensity']) * self.to_msun
            min_max = [-20, 20]
            dmd = vis.binned_stats_img(dm_coords, np.log(np.array(dm_density)), min_max)
            extent = [min_max[0], min_max[1], min_max[0], min_max[1]]
            return dmd, extent

    def center_coordinates(self, coordinates, science_data):
        center_pos = ast.literal_eval(str(science_data["center_position"]))
        return (np.array(coordinates, dtype=np.float64) - center_pos) * self.dist_units_kpc

