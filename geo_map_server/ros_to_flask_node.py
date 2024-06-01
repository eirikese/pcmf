#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import requests
import json
import numpy as np

class PathToMapNode:
    def __init__(self):
        rospy.init_node('path_to_map_node', anonymous=True)
        self.sub = rospy.Subscriber('/pcmf_anchor_pivot_node/transformed_path', PointCloud2, self.callback)
        # self.sub = rospy.Subscriber('/reference_map', PointCloud2, self.callback)
        self.base_lat = 63.4393000
        self.base_lon = 10.3990500
        self.earth_radius_km = 6371.0088
        self.flask_server_url = 'http://localhost:5000/update'

    def point_cloud_to_lat_lon(self, cloud):
        path_points = []
        for point in point_cloud2.read_points(cloud, skip_nans=True, field_names=("x", "y")):
            delta_lat = (point[1] / 1000) / (self.earth_radius_km * np.pi / 180)
            delta_lon = (point[0] / 1000) / (self.earth_radius_km * np.pi / 180) / np.cos(self.base_lat * np.pi / 180)
            lat = self.base_lat + delta_lat
            lon = self.base_lon + delta_lon
            path_points.append({'lat': lat, 'lon': lon})
        return path_points

    def send_data_to_flask(self, path_points):
        try:
            response = requests.post(self.flask_server_url, json={'path_points': path_points})
            rospy.loginfo("Data sent to Flask server, response: %s", response.text)
        except requests.exceptions.RequestException as e:
            rospy.logerr("Request failed: %s", e)

    def callback(self, data):
        path_points = self.point_cloud_to_lat_lon(data)
        self.send_data_to_flask(path_points)

if __name__ == '__main__':
    try:
        node = PathToMapNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
