#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import folium
import numpy as np

class PathToMapNode:
    def __init__(self):
        rospy.init_node('path_to_map_node', anonymous=True)

        # Subscriber to the PointCloud2 topic
        self.sub = rospy.Subscriber('/pcmf_anchor_pivot_node/transformed_path', PointCloud2, self.callback)

        # Store the base lat/long for conversion
        self.base_lat = 63.439274
        self.base_lon = 10.3990245

        # Earth's radius in kilometers at the given latitude (approximation)
        self.earth_radius_km = 6371.0088
        # List to store converted lat/long points
        self.path_points = []

    def point_cloud_to_lat_lon(self, cloud):
        # Convert Cartesian (x, y) in the PointCloud to lat/long
        for point in point_cloud2.read_points(cloud, skip_nans=True, field_names=("x", "y")):
            # Convert from meters to geographical coordinates
            # Assuming x=0, y=0 matches base_lat, base_lon
            delta_lat = (point[1] / 1000) / (self.earth_radius_km * np.pi/180)
            delta_lon = (point[0] / 1000) / (self.earth_radius_km * np.pi/180) / np.cos(self.base_lat * np.pi/180)
            lat = self.base_lat + delta_lat
            lon = self.base_lon + delta_lon
            self.path_points.append((lat, lon))

    def plot_and_save_map(self):
        # Create a folium map centered around the first point in the path
        if self.path_points:
            m = folium.Map(location=self.path_points[0], zoom_start=15, tiles='OpenStreetMap')
            
            # Add points to the map
            for lat, lon in self.path_points:
                folium.CircleMarker(location=[lat, lon], radius=1, color='red').add_to(m)
            
            # Save the map
            m.save('/home/mr_fusion/lidar_ws/src/pcmf/maps/path_map.html')
            rospy.loginfo("Map saved to path_map.html")

    def callback(self, data):
        # Clear previous path points to store new ones
        self.path_points = []

        # Convert PointCloud2 points to lat/lon and store them
        self.point_cloud_to_lat_lon(data)

        # Plot and save the map as an HTML file
        self.plot_and_save_map()

        # Shutdown the node
        rospy.signal_shutdown("Map saved, shutting down.")

if __name__ == '__main__':
    try:
        path_to_map_node = PathToMapNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
