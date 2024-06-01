import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def pcd_to_npy(pcd_file, npy_file, voxel_size, z_min, z_max):
    # Load the PCD file
    pcd = o3d.io.read_point_cloud(pcd_file)
    print("n dim pcd: ", np.asarray(pcd.points).shape)
    print(f"Number of points before voxel down-sampling: {len(pcd.points)}")

    # Perform voxel down-sampling
    pcd = pcd.voxel_down_sample(voxel_size)
    print(f"Number of points after voxel down-sampling: {len(pcd.points)}")

    # Convert to numpy array
    points = np.asarray(pcd.points)

    # Filter points based on Z value
    points = points[(points[:, 2] >= z_min) & (points[:, 2] <= z_max)]     # toggle this line to enable/disable z-filtering

    # Keep only the X and Y coordinates
    points_xy = points[:, :2] * 1/100 # scaling for pcmf_icp_npy 1:100

    # Save to .npy file
    print("n dimensions in saved .npy: ", points.shape)
    np.save(npy_file, points) # change to points_xy to save only x and y coordinates

# Replace 'input.pcd' and 'output.npy' with your actual file paths
# Adjust voxel_size to achieve the desired downsampling.
# Adjust z_min and z_max to filter points based on Z value.
import_pcd = r'C:\Users\eirik\OneDrive - NTNU\Semester 10\code\pcmf\maps\mr_2710_complete.pcd'
export_npy = r'C:\Users\eirik\OneDrive - NTNU\Semester 10\code\pcmf\maps\mr_2710_complete.npy'
pcd_to_npy(import_pcd, export_npy, voxel_size=0.8, z_min=-10.0, z_max=100.0)

# plot the point cloud
pcd = np.load(export_npy)
print(pcd.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], s=0.1)
ax.set_aspect('equal')
plt.show()
