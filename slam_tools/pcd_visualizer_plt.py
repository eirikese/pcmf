import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def visualize_pcd(file_path):
    # Load the PCD file
    pcd = o3d.io.read_point_cloud(file_path)

    # Convert to numpy array
    points = np.asarray(pcd.points)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)

    # Setting the title with file name and number of points
    title = f"{file_path} - Points: {len(points)}"
    ax.set_title(title)

    # Show the plot
    plt.show()

# Replace 'your_file.pcd' with the path to your PCD file
visualize_pcd(r"C:\Users\eirik\OneDrive - NTNU\Semester 10\code\pcmf\slam_tools\hdl_mr_map.pcd")
