#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header

# Load .npy file function for loading numpy arrays
def load_npy_file(file_path):
    return np.load(file_path)

# Convert numpy array to PointCloud2
def numpy_to_pointcloud2(arr, frame_id="map"):
    header = Header(frame_id=frame_id)
    header.stamp = rospy.Time.now()
    # Ensure arr has only three columns (x, y, z)
    if arr.shape[1] != 3:
        # Assume z-coordinate is missing and append zero z-coordinates
        zeros = np.zeros((arr.shape[0], 1))
        arr = np.hstack((arr, zeros))
    return pc2.create_cloud_xyz32(header, arr)

# Main publisher function
def publisher():
    rospy.init_node('reference_points_publisher', anonymous=True)
    reference_map_pub = rospy.Publisher('reference_map', PointCloud2, queue_size=10)
    reference_points_pub = rospy.Publisher('reference_corners', PointCloud2, queue_size=10)
    rate = rospy.Rate(10) # 10hz

    # Load points from .npy files
    corners_data = load_npy_file('/home/eirik/lidar_ws/src/pcmf/maps/selected_points.npy')
    reference_data = load_npy_file('/home/eirik/lidar_ws/src/pcmf/maps/all_points.npy')

    # Ensure data is 3D
    if corners_data.shape[1] == 2:
        zeros = np.zeros((corners_data.shape[0], 1))
        corners_data = np.hstack((corners_data, zeros))
    if reference_data.shape[1] == 2:
        zeros = np.zeros((reference_data.shape[0], 1))
        reference_data = np.hstack((reference_data, zeros))

    # Convert numpy arrays to PointCloud2
    corners_pointcloud = numpy_to_pointcloud2(corners_data)
    reference_pointcloud = numpy_to_pointcloud2(reference_data)

    # shift both clouds 200 left
    corners_data[:, 0] -= 200
    reference_data[:, 0] -= 200

    # # Reverse corner points order in list
    # corners_data = corners_data[::-1]

    while not rospy.is_shutdown():
        reference_map_pub.publish(reference_pointcloud)
        reference_points_pub.publish(corners_pointcloud)
        rate.sleep()

if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass
