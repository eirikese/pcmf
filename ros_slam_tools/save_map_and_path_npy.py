#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import ros_numpy

# Flags to check if data has been saved
path_points_saved = False
map_points_processed_saved = False

def path_points_callback(data):
    global path_points_saved
    if not path_points_saved:
        # Convert PointCloud2 to numpy array and save
        np_data = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(data)
        np.save('/home/mr_fusion/lidar_ws/src/pcmf/maps/mr_2710_path_processed.npy', np_data)
        path_points_saved = True
        check_shutdown()

def map_points_processed_callback(data):
    global map_points_processed_saved
    if not map_points_processed_saved:
        # Convert PointCloud2 to numpy array and save
        np_data = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(data)
        np.save('/home/mr_fusion/lidar_ws/src/pcmf/maps/mr_2710_map_processed.npy', np_data)
        map_points_processed_saved = True
        check_shutdown()

def check_shutdown():
    # Shut down the node if data from both topics is saved
    if path_points_saved and map_points_processed_saved:
        rospy.signal_shutdown("Data saved from both topics.")

def listener():
    rospy.init_node('data_saver_node', anonymous=True)
    rospy.Subscriber("/hdl_graph_slam/path_points", PointCloud2, path_points_callback)
    rospy.Subscriber("/map_points_processed", PointCloud2, map_points_processed_callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
