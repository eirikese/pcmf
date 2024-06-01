#!/usr/bin/env python

import rospy
import numpy as np
import ezdxf
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from sklearn.linear_model import RANSACRegressor
from math import atan2, degrees

# Load DXF file function
def load_dxf_file(file_path):
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()
    lines = [entity for entity in msp if entity.dxftype() == 'LINE']

    point_list = []
    for line in lines:
        start = np.array([line.dxf.start.x, line.dxf.start.y, 0])  # z-coordinate set to 0
        end = np.array([line.dxf.end.x, line.dxf.end.y, 0])        # z-coordinate set to 0
        length = np.linalg.norm(end[:2] - start[:2])
        num_samples = max(int(length) + 1, 2)  # Ensuring at least two samples per line

        for i in range(num_samples):
            point = start + (end - start) * i / (num_samples - 1)
            point_list.append(point)

    return np.array(point_list)

# Convert numpy array to PointCloud2
def numpy_to_pointcloud2(arr, frame_id="map"):
    header = Header(frame_id=frame_id)
    header.stamp = rospy.Time.now()
    # Ensure arr has only three columns (x, y, z)
    if arr.shape[1] != 3:
        raise ValueError("Array must be Nx3 dimensions.")
    return pc2.create_cloud_xyz32(header, arr)

# Calculate the orientation of a line segment
def calculate_orientation(line):
    dx = line[-1, 0] - line[0, 0]
    dy = line[-1, 1] - line[0, 1]
    return atan2(dy, dx)

# Check if two lines are parallel within a specified angular threshold
def are_parallel(orientation1, orientation2, angle_threshold):
    return abs(degrees(orientation1) - degrees(orientation2)) < angle_threshold

# Compute the distance between two lines
def line_distance(line1, line2):
    mid1 = np.mean(line1, axis=0)
    mid2 = np.mean(line2, axis=0)
    return np.linalg.norm(mid1 - mid2)

# Function to merge lines that are parallel and close to each other
def merge_lines(lines, orientations, line_dist_threshold, angle_threshold):
    merged_lines = []
    merged = [False] * len(lines)

    for i in range(len(lines)):
        if merged[i]:
            continue

        for j in range(i + 1, len(lines)):
            if merged[j]:
                continue

            if are_parallel(orientations[i], orientations[j], angle_threshold) and \
               line_distance(lines[i], lines[j]) < line_dist_threshold:
                merged_lines.append(np.concatenate((lines[i], lines[j])))
                merged[i] = merged[j] = True
                break

        if not merged[i]:
            merged_lines.append(lines[i])

    return merged_lines

# Detect corners using RANSAC and process the data
def detect_corners(npy_data):
    model = RANSACRegressor(residual_threshold=1, max_trials=100)
    edges = []
    end_points = []

    while len(npy_data) > 2:
        model.fit(npy_data[:, 0].reshape(-1, 1), npy_data[:, 1])
        inlier_mask = model.inlier_mask_
        outliers = npy_data[~inlier_mask]
        inliers = npy_data[inlier_mask]

        if len(inliers) > 1:
            edges.append(inliers)
            sorted_inliers = inliers[inliers[:, 0].argsort()]
            end_points.extend([sorted_inliers[0], sorted_inliers[-1]])

        npy_data = outliers

        if len(npy_data) < 0.05 * len(npy_data):
            break

    orientations = [calculate_orientation(edge) for edge in edges]
    merged_lines = merge_lines(edges, orientations, line_dist_threshold=2, angle_threshold=10)

    merged_end_points = []
    for line in merged_lines:
        merged_end_points.extend([line[0], line[-1]])

    return merged_end_points

# Main publisher function
def publisher():
    rospy.init_node('reference_map_dxf_publisher', anonymous=True)
    map_pub = rospy.Publisher('reference_map', PointCloud2, queue_size=10)
    corners_pub = rospy.Publisher('reference_corners', PointCloud2, queue_size=10)
    rate = rospy.Rate(10) # 10hz

    dxf_data = load_dxf_file('/home/eirik/lidar_ws/src/pcmf/maps/bridge_map.dxf')
    map_pointcloud = numpy_to_pointcloud2(dxf_data)

    corners = detect_corners(dxf_data)
    corners_pointcloud = numpy_to_pointcloud2(np.array(corners))

    while not rospy.is_shutdown():
        map_pub.publish(map_pointcloud)
        corners_pub.publish(corners_pointcloud)
        rate.sleep()

if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass
