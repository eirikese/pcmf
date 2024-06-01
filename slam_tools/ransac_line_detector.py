import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics.pairwise import euclidean_distances
from math import atan2, degrees

# Load the point cloud data
point_cloud = np.load('/home/mr_fusion/lidar_ws/src/pcmf/maps/reference_map_mr_partial.npy') * 100
# point_cloud = np.load('/home/mr_fusion/lidar_ws/src/pcmf/maps/mr_2710_complete.npy')


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

    return merged

# Function to perform Sequential RANSAC
def sequential_ransac(points, threshold=1, max_trials=100, merge_dist=6.0, line_dist_threshold=1, angle_threshold=10):
    model = RANSACRegressor(residual_threshold=threshold, max_trials=max_trials)
    edges = []
    end_points = []

    while len(points) > 0:
        model.fit(points[:, 0].reshape(-1, 1), points[:, 1])
        inlier_mask = model.inlier_mask_
        outliers = points[~inlier_mask]
        inliers = points[inlier_mask]
        
        if len(inliers) > 0:
            edges.append(inliers)
            sorted_inliers = inliers[inliers[:, 0].argsort()]
            end_points.extend([sorted_inliers[0], sorted_inliers[-1]])
        
        points = outliers

        if len(points) < 0.05 * len(point_cloud):
            break

    # Calculate orientations and merge parallel and close lines
    orientations = [calculate_orientation(edge) for edge in edges]
    merged_lines = merge_lines(edges, orientations, line_dist_threshold, angle_threshold)

    # Merge close end points for merged lines
    merged_end_points = []
    for line in merged_lines:
        merged_end_points.extend([line[0], line[-1]])
    final_merged_points = merge_end_points(merged_end_points, merge_dist)

    return merged_lines, final_merged_points

# Run Sequential RANSAC
detected_edges, merged_end_points = sequential_ransac(point_cloud)

# Plot the results
plt.figure(figsize=(10, 10))
for edge in detected_edges:
    plt.scatter(edge[:, 0], edge[:, 1], alpha=0.7)
for point in merged_end_points:
    plt.scatter(point[0], point[1], color='red', s=100)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Detected Edges with Merged End Points')
plt.axis('equal')  
plt.gca().set_aspect('equal')
plt.show()
