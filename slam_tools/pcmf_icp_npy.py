'''This code provides a comprehensive implementation of the Iterative Closest Point (ICP) algorithm for point cloud registration using the Open3D library. 
It is designed to align two point clouds (source and target), which are loaded from .npy files. The main functionalities of the code are:

File Imports and Initial Transformations:
    The code begins by importing source and target point clouds from .npy files.
    An initial transformation (rotation and translation) is applied to the target point cloud.
ICP Registration:
    The ICP algorithm is implemented with parameters like maximum rotation increments and a threshold for the Root Mean Square Error (RMSE).
    The code incrementally rotates the source point cloud to align it with the target, evaluating the alignment quality using the RMSE.
Visualization:
    The code utilizes matplotlib for visualizing the point clouds before and after the ICP registration.
    It displays initial and final positions of the point clouds with details about the transformation (offset and angle).
Transformation and Result Analysis:
    After the registration process, the code extracts the translation and rotation results from the transformation matrix.
    It prints the best transformation matrix obtained from the ICP algorithm.
Utility Functions:
    apply_transformation: Applies rotation and translation to points.
    get_rotation_translation_from_matrix: Extracts rotation and translation from a transformation matrix.
Main Function:
    Orchestrates the entire process from loading data, applying transformations, 
    running ICP, visualizing results, and printing the final transformation matrix.'''

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# File imports
# target_file = r"C:\Users\eirik\OneDrive - NTNU\Semester 9\Prosjektoppgave\o3d_map\bay_template_data\data_0.npy"  # Path to the source .npy file
source_file = r"C:\Users\eirik\OneDrive - NTNU\Semester 10\code\pcmf\maps\mr_2710_complete.npy"  # Path to the source .npy file
target_file = r"C:\Users\eirik\OneDrive - NTNU\Semester 9\Prosjektoppgave\o3d_map\bay_evaluation_data\data_0.npy"  # Path to the target .npy file

# Initial transformation
rotation_angle = np.radians(100) # SET TARGET ROTATION ANGLE
xy_translation = [10, 10] # SET TARGET TRANSLATION

# ICP parameters
rmse_threshold = 0.01
rotation_increment = np.radians(10)
max_rotation_increments = 2*np.pi/rotation_increment # adds up to one full rotation

time_plot_open = 10 # Seconds

def apply_transformation(points, translation=np.array([0, 0]), rotation_angle=0):
    # Define a 2D rotation matrix
    c, s = np.cos(rotation_angle), np.sin(rotation_angle)
    R = np.array([[c, -s], [s, c]])
    
    # Apply rotation and then translation
    transformed_points = np.dot(points, R) + translation
    
    return transformed_points

def get_rotation_translation_from_matrix(matrix):
    # Assuming matrix is a 4x4 homogeneous transformation matrix
    rotation_matrix = matrix[:3, :3]
    translation_vector = matrix[:3, 3]

    # Calculate the Euler angles from the rotation matrix
    sy = np.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] +  rotation_matrix[1,0] * rotation_matrix[1,0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
        y = np.arctan2(-rotation_matrix[2,0], sy)
        z = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
    else:
        x = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])
        y = np.arctan2(-rotation_matrix[2,0], sy)
        z = 0

    euler_angles = np.array([x, y, z])
    return euler_angles, translation_vector

def ssa(angle_deg):
    if abs(angle_deg) > 180:
        return abs(abs(angle_deg) - 360)
    return abs(angle_deg)
    
def main():

    # Load points from .npy files
    source_points = np.load(source_file)
    target_points = np.load(target_file)
    target_points = apply_transformation(target_points, translation=np.array(xy_translation), rotation_angle=rotation_angle)

    # Convert to Open3D point cloud objects
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(np.hstack((source_points, np.zeros((source_points.shape[0], 1)))))
    source2 = source # keep original source cloud for final visualization
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(np.hstack((target_points, np.zeros((target_points.shape[0], 1)))))

    # Calculate centroids of source and target
    source_centroid = np.mean(np.asarray(source.points), axis=0)
    target_centroid = np.mean(np.asarray(target.points), axis=0)
    translation_to_target_centroid = target_centroid - source_centroid

    # Initialize the transformation matrix
    trans_init = np.identity(4)
    trans_init[0:3, 3] = translation_to_target_centroid[0:3]

    # Visualize initial point clouds using plt
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    plt.subplots_adjust(top=0.75)

    initial_offset_text = f'Initial Offset: {xy_translation}, Angle: {ssa(np.degrees(rotation_angle)):.2f}°'
    axs[0].set_title('Before ICP Registration\n' + initial_offset_text)
    axs[0].scatter(np.asarray(source.points)[:, 0], np.asarray(source.points)[:, 1], c='r', label='Source', s=3)
    axs[0].scatter(np.asarray(target.points)[:, 0], np.asarray(target.points)[:, 1], c='g', label='Target', s=3)
    axs[0].legend()
    axs[0].axis('equal')
    
    # Initialize the RMSE threshold and rotation increment
    best_icp_result = None
    best_rmse = float('inf')
    increment_count = 0

    # Run the ICP registration until the RMSE is below the threshold or the source has been fully rotated
    while True and increment_count < max_rotation_increments:
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, 0.99, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        # Update the best ICP result if the current RMSE is lower than the previous best
        if reg_p2p.inlier_rmse < best_rmse:
            best_icp_result = reg_p2p
            best_rmse = reg_p2p.inlier_rmse
            best_fit_increment = increment_count

        # Check if RMSE is below the threshold
        if best_rmse <= rmse_threshold and not best_rmse == 0:
            break

        # Rotate the source point cloud by the increment
        source.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, 0, rotation_increment)), center=source_centroid)
        # Update the initial transformation to be used as an initial guess for the next iteration
        trans_init[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, rotation_increment))
        increment_count += 1

    # Extract translation and rotation from the transformation matrix
    transformation = best_icp_result.transformation
    translation_result = transformation[0:3, 3]
    rotation_result = transformation[0:3, 0:3]
    rotation_angle_result = np.arccos((np.trace(rotation_result) - 1) / 2)  + best_fit_increment * rotation_increment
    transformation_text = f'Result Offset: {translation_result[:2]}, Angle: {ssa(np.degrees(rotation_angle_result)):.2f}°'
    
    # After finding the best ICP transformation, apply it to the source. First incremental, then icp transform
    source2.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, 0, rotation_increment*best_fit_increment)), center=source_centroid)
    source2.transform(best_icp_result.transformation)

    # Visualize the point clouds after registration using plt
    axs[1].set_title('After ICP Registration\n' + transformation_text)
    axs[1].scatter(np.asarray(source2.points)[:, 0], np.asarray(source2.points)[:, 1], c='r', label='Source (Transformed)', s=3)
    axs[1].scatter(np.asarray(target.points)[:, 0], np.asarray(target.points)[:, 1], c='g', label='Target', s=3)
    axs[1].legend()
    axs[1].axis('equal')
   
    # Save and show the plots
    plot_stats = "   Inlier RMSE: " + str(reg_p2p.inlier_rmse)  + "\nRotations performed: " + str(increment_count) + "*" + str(np.degrees(rotation_increment)) + "°" + "    Best fit increment no: " + str(best_fit_increment) + " / " + str(increment_count) 
    fig.suptitle('Incremental Rotation Centroid-ICP Point Cloud Fitting, Downscaled 1:100\n' + plot_stats, fontsize=16)
    # plt.savefig("o3d_npy_centroid_icp.png")
    plt.tight_layout()
    plt.show()
#    plt.show(block=False)
 #   plt.pause(time_plot_open) # Keep open for 5 seconds
  #  plt.close()    
    
    print("Best transformation:")
    print(best_icp_result.transformation)

if __name__ == "__main__":
    main()
    