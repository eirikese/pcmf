import open3d as o3d
import numpy as np

def visualize_pcd_with_open3d(file_path, z_min=None, z_max=None):
    # Load the PCD file
    pcd = o3d.io.read_point_cloud(file_path)

    # Filter points based on z_min and z_max
    if z_min is not None or z_max is not None:
        points = np.asarray(pcd.points)
        indices = []
        for i, point in enumerate(points):
            if z_min is not None and point[2] < z_min:
                continue
            if z_max is not None and point[2] > z_max:
                continue
            indices.append(i)
        pcd = pcd.select_by_index(indices)

    # Print the number of points and file name
    print(f"File: {file_path}")
    print(f"Number of points: {len(pcd.points)}")
    
    # Paint the point cloud in red
    # pcd.paint_uniform_color([0, 0, 0])

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

# Replace 'your_file.pcd' with the path to your PCD file
# pcd_file_path = r"C:\Users\eirik\OneDrive - NTNU\Semester 10\code\pcmf\slam_tools\hdl_mr_map.pcd"
pcd_file_path = r"/home/mr_fusion/lidar_ws/src/pcmf/maps/mr_2710_complete.pcd"
visualize_pcd_with_open3d(pcd_file_path, z_min=-5.0, z_max=5.0)