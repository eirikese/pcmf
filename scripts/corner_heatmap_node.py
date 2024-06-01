#!/usr/bin/env python

import rospy
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

# Global variable to store corner positions
corner_positions = []

def corner_points_callback(msg):
    global corner_positions

    # Extract corner positions from PointCloud2 message
    points = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)

    for point in points:
        x, y, z = point
        corner_positions.append((x, y))

def shutdown_hook():
    global corner_positions

    if corner_positions:
        corner_positions = np.array(corner_positions)
        x = corner_positions[:, 0]
        y = corner_positions[:, 1]

        # Calculate the histogram
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=100)

        # Normalize the heatmap
        heatmap = heatmap / np.sum(heatmap)

        # Padding: Expand the x and y ranges pad %
        x_pad = (xedges[-1] - xedges[0]) * 0.1
        y_pad = (yedges[-1] - yedges[0]) * 0.1

        # Adjusted extent for the heatmap
        extent = [xedges[0] - x_pad, xedges[-1] + x_pad, yedges[0] - y_pad, yedges[-1] + y_pad]

        # Plotting
        plt.figure(figsize=(8, 8))
        plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='viridis') 
        plt.colorbar()
        plt.axis('equal')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title('Corner Estimate Heatmap (XY Plane)')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')

        # Save the heatmap as a PDF with a timestamp
        timestamp = rospy.Time.now().to_sec()
        file_name = f'/home/mr_fusion/lidar_ws/src/pcmf/plots/corner_heatmap_{timestamp}.pdf'
        plt.savefig(file_name, format='pdf')
        rospy.loginfo(f'Corner heatmap saved as {file_name}')

if __name__ == '__main__':
    rospy.init_node('corner_heatmap_node')
    rospy.loginfo('corner_heatmap_node started,subscribe to /pcmf_icp_pcmf_node/detected_corners')
    
    # Subscribe to corner points topic
    rospy.Subscriber('/pcmf_icp_pcmf_node/detected_corners', PointCloud2, corner_points_callback)

    # Register shutdown hook to generate and save the heatmap
    rospy.on_shutdown(shutdown_hook)

    rospy.spin()
