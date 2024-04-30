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





class IllustrisSdssDatasetMultidim(IllustrisSdssDatasetWithMetadata):
    def __init__(
            self,
            data_directories: list[str],
            info_dir: str,
            data_aspect: str = None,
            extension: str = "fits",
            minsize: int = 100,
            transform=None,
    ):
        # Superclass constructor call will find FITS files (relevant for training) and store them
        super().__init__(data_directories, extension, minsize, transform)

        # We extend the constructor to store The h5 files with additional info
        self.hdf_files = {}
        for data_directory in data_directories:
            for file in sorted(os.listdir(data_directory)):
                if file.endswith('hdf5'):
                    fname, _ = file.split('.')
                    _, subhalo_id = fname.split('_')
                    hdf_filename = os.path.join(data_directory, file)
                    self.hdf_files[subhalo_id] = hdf_filename

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
        h5_path = self.hdf_files[subhalo_id]
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

    def make_stars(self, science_data, cutout):
        if cutout.keys().__contains__(TngParticleTypes.STARS):
            stars = cutout[TngParticleTypes.STARS]
            star_coords = self.center_coordinates(stars['Coordinates'], science_data)
            #radius = np.array(stars['StellarHsml']) * self.dist_units_kpc
            radius = np.ones(star_coords.shape[0]) * 10**(-11)

            # Only stars, no star winds
            mask = np.where(np.array(stars['GFM_StellarFormationTime']) > 0)
            return vis.star_point_cloud(star_coords[mask], radius[mask])

    def make_gas_temperature_field(self, science_data, cutout):
        if cutout.keys().__contains__(TngParticleTypes.GAS):
            gas = cutout[TngParticleTypes.GAS]
            gas_coords = self.center_coordinates(gas['Coordinates'], science_data)
            temperature = self.gas_temperature(gas)
            min_max = [-20, 20]
            t = vis.binned_stats_img(gas_coords, np.log10(temperature), min_max)
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

    def make_particle_clouds(self, science_data, cutout):
        pointclouds = {}
        if cutout.keys().__contains__(TngParticleTypes.GAS):
            particles = cutout[TngParticleTypes.GAS]
            gas_coords = self.center_coordinates(particles['Coordinates'], science_data)
            attribute_keys = ['temperature', 'potential', 'metallicity', 'velocity']
            cmaps = ['hot', 'magma', 'viridis', 'turbo']
            gas_attributes = [self.gas_temperature(particles),
                              np.array(particles['Potential'], dtype=np.float64),
                              np.array(particles['GFM_Metallicity'], dtype=np.float64),
                              np.array(particles['SubfindVelDisp'], dtype=np.float64)]
            pointclouds['gas'] = self.make_pointcloud(gas_coords, attribute_keys, gas_attributes, cmaps)
        if cutout.keys().__contains__(TngParticleTypes.DM):
            particles = cutout[TngParticleTypes.DM]
            dm_coords = self.center_coordinates(particles['Coordinates'], science_data)
            attribute_keys = ['potential', 'density', 'velocity']
            cmaps = ['magma', 'jet', 'turbo']
            dm_attributes = [ np.array(particles['Potential'], dtype=np.float64),
                              np.array(particles['SubfindDensity'], dtype=np.float64),
                              np.array(particles['SubfindVelDisp'], dtype=np.float64)]
            pointclouds['dm'] = self.make_pointcloud(dm_coords, attribute_keys,dm_attributes, cmaps)
        if cutout.keys().__contains__(TngParticleTypes.STARS):
            particles = cutout[TngParticleTypes.STARS]
            star_coords = self.center_coordinates(particles['Coordinates'], science_data)
            attribute_keys = ['mass', 'metallicity']
            cmaps = ['gnuplot', 'viridis']
            star_attributes = [np.array(particles['Masses'], dtype=np.float64),
                               np.array(particles['GFM_Metallicity'], dtype=np.float64)]
            pointclouds['stars'] = self.make_pointcloud(star_coords, attribute_keys, star_attributes, cmaps)
        return pointclouds

    def make_pointcloud(self, coordinates, keys, attributes, cmaps):
        pointcloud_map = {}
        for i in range(len(attributes)):
            pointcloud_map[keys[i]] = vis.make_pointcloud(coordinates, attributes[i], cmap=cmaps[i])
        return pointcloud_map
    def make_gas_particle_cloud_bin(self, science_data, cutout):
        particle_dict = {}
        if cutout.keys().__contains__(TngParticleTypes.GAS):
            particles = cutout[TngParticleTypes.GAS]
            particle_dict['coordinates'] = self.center_coordinates(particles['Coordinates'], science_data).tolist()
            particle_dict['temperature'] = self.gas_temperature(particles).tolist()
            pot = np.array(particles['Potential'], dtype=np.float64)
            pot /= pot.mean()
            particle_dict["potential"] = pot.tolist()
            particle_dict['metallicity'] = np.array(particles['GFM_Metallicity'], dtype=np.float64).tolist()
            particle_dict['velocity'] = np.array(particles['SubfindVelDisp']).tolist()
        return particle_dict



    def center_coordinates(self, coordinates, science_data):
        center_pos = ast.literal_eval(str(science_data["center_position"]))
        return (np.array(coordinates) - center_pos) * self.dist_units_kpc

    def gas_temperature(self, gas):
        # calculate the temperature because apparently, physicians love to do the math by themselves...
        u = np.array(gas["InternalEnergy"], dtype=np.float64) * 100000. ** 2  # CGS Units
        x_e = np.array(gas["ElectronAbundance"], dtype=np.float64)
        x_h = 0.76
        gamma = 5 / 3
        k_b = 1.380649 * 10 ** (-16) # CGS Units
        mass = np.array(gas["Masses"], dtype=np.float64) * self.to_msun * 2. * 10**33 # First to Sun Masses, then to gram
        mu = (4. / (1. + 3. * x_h + 4. * x_h * x_e)) * mass
        temp = np.array((gamma - 1.) * (u / k_b) * mu, dtype=np.float64)
        return np.log10(temp)


