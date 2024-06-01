# convert 2d npy to pcd from predefined path without parsing arguments

import numpy as np
import open3d as o3d

# load npy in two dimensions
npy_path = '/home/mr_fusion/lidar_ws/src/pcmf/maps/reference_map_mr_partial.npy'
npy = np.load(npy_path)
npy = npy.astype(np.float32)

# convert to pcd
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(npy)
o3d.io.write_point_cloud('/home/mr_fusion/lidar_ws/src/pcmf/maps/reference_map_mr_partial.pcd', pcd)
