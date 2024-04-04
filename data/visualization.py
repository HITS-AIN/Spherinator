import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as c
import open3d as o3d
from matplotlib import cm
from scipy.stats import binned_statistic_2d

def binned_stats_img(coordinates, value, min_max):
    img, _, _, _ = binned_statistic_2d(coordinates[:,0], coordinates[:,1], value,
                                       'sum', bins=[128, 128], range=[min_max, min_max])
    return img

def circle_mask(coords, values, radius):
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    new_x, new_y, new_z, new_values = [], [], [], []
    for i in range(len(values)):
        distance = np.sqrt((x[i] ** 2) + (y[i] ** 2) + (z[i] ** 2))
        if distance <= radius:
            new_x.append(x[i])
            new_y.append(y[i])
            new_z.append(z[i])
            new_values.append(values[i])
    return np.array(new_x), np.array(new_y), np.array(new_z), np.array(new_values)


def make_pointcloud(coords, potential, mask_radius=100.):
    # Circle mask
    x, y, z, values = circle_mask(coords, potential, mask_radius)
    gas_cloud = o3d.geometry.PointCloud()
    gas_cloud.points = o3d.utility.Vector3dVector(np.column_stack((x, y, z)))

    cmap = plt.get_cmap('magma')
    norm = c.Normalize(vmin=values.min(), vmax=values.max())
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    rgba = scalar_map.to_rgba(values)[:, :3]
    gas_cloud.colors = o3d.utility.Vector3dVector(rgba)
    return gas_cloud


def star_point_cloud(star_coords, radius, mask_radius=100.):
    x, y, z, radius = circle_mask(star_coords, radius, mask_radius)
    stars = o3d.geometry.PointCloud()
    #for i in range(len(radius)):
    #    star = o3d.geometry.TriangleMesh.create_sphere(radius=radius[i])
    #    star.translate(np.array([x[i], y[i], z[i]]))
    #    pc_star = star.sample_points_uniformly(number_of_points=500)
    #    stars.points.extend(pc_star.points)
    stars.points = o3d.utility.Vector3dVector(np.column_stack((x, y, z)))
    color = np.ones((len(stars.points), 3))
    stars.colors = o3d.utility.Vector3dVector(color)
    return stars



