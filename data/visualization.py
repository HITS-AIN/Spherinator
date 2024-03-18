import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as c
import open3d as o3d
from matplotlib import cm


def gas_potential_pointcloud(gas_coords, potential):
    # Circle mask
    radius = 100.
    x = gas_coords[:, 0]
    y = gas_coords[:, 1]
    z = gas_coords[:, 2]
    new_x, new_y, new_z, new_values = [], [], [], []
    for i in range(len(potential)):
        distance = np.sqrt((x[i] ** 2) + (y[i] ** 2) + (z[i] ** 2))
        if distance <= radius:
            new_x.append(x[i])
            new_y.append(y[i])
            new_z.append(z[i])
            new_values.append(potential[i])
    x, y, z, values = np.array(new_x), np.array(new_y), np.array(new_z), np.array(new_values)
    gas_cloud = o3d.geometry.PointCloud()
    gas_cloud.points = o3d.utility.Vector3dVector(np.column_stack((x, y, z)))

    cmap = plt.get_cmap('magma')
    norm = c.Normalize(vmin=values.min(), vmax=values.max())
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    rgba = scalar_map.to_rgba(values)[:, :3]
    gas_cloud.colors = o3d.utility.Vector3dVector(rgba)
    return gas_cloud

