#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics.pairwise import euclidean_distances
from math import atan2, degrees

# Load .npy file function
def load_npy_file(file_path):
    return np.load(file_path)

# Convert numpy array to PointCloud2
def numpy_to_pointcloud2(arr, frame_id="map"):
    header = Header(frame_id=frame_id)
    header.stamp = rospy.Time.now()
    return pc2.create_cloud_xyz32(header, arr)

# Function to calculate the orientation of a line segment
def calculate_orientation(line):
    dx = line[-1, 0] - line[0, 0]
    dy = line[-1, 1] - line[0, 1]
    return atan2(dy, dx)

# Function to check if two lines are parallel within a specified angular threshold
def are_parallel(orientation1, orientation2, angle_threshold):
    return abs(degrees(orientation1) - degrees(orientation2)) < angle_threshold

# Function to compute the distance between two lines
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

# Function to merge close end points
def merge_end_points(points, dist_threshold):
    if not points:
        return []

    distances = euclidean_distances(points)
    np.fill_diagonal(distances, np.inf)

    merged = []
    visited = set()

    for i, point in enumerate(points):
        if i in visited:
            continue

        close_points = [j for j in range(len(points)) if distances[i, j] < dist_threshold]

        if close_points:
            avg_point = np.mean([points[j] for j in close_points + [i]], axis=0)
            merged.append(avg_point)
            visited.update(close_points)
            visited.add(i)

        # add single points
        if i not in visited:
            merged.append(point)
    return merged

# Function to detect corners using RANSAC and merge end points
def detect_corners(npy_data):
    model = RANSACRegressor(residual_threshold=1, max_trials=100)
    edges = []
    end_points = []

    while len(npy_data) > 2:
        model.fit(npy_data[:, 0].reshape(-1, 1), npy_data[:, 1])
        inlier_mask = model.inlier_mask_
        outliers = npy_data[~inlier_mask]
        inliers = npy_data[inlier_mask]

        if len(inliers) > 0:
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

    final_merged_points = merge_end_points(merged_end_points, dist_threshold=10)

    return final_merged_points
    # return merged_end_points

# Main publisher function
def publisher():
    rospy.init_node('reference_map_npy_publisher', anonymous=True)
    map_pub = rospy.Publisher('reference_map', PointCloud2, queue_size=10)
    corners_pub = rospy.Publisher('reference_corners', PointCloud2, queue_size=10)
    rate = rospy.Rate(10) # 10hz

    npy_data = load_npy_file('/home/eirik/lidar_ws/src/pcmf/maps/bay_lines_v3.npy')
    # npy_data = load_npy_file('/home/eirik/lidar_ws/src/pcmf/maps/reference_map_mr_complete.npy') * 100
    
    # Shift map and corners 200 units to the left
    npy_data[:, 0] -= 200

    if npy_data.ndim == 2:
        zeros = np.zeros((npy_data.shape[0], 1))
        npy_data = np.hstack((npy_data, zeros))

    npy_data = npy_data # * 100
    map_pointcloud = numpy_to_pointcloud2(npy_data)

    # Detect corners and publish them
    corners = detect_corners(npy_data)
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
