#!/usr/bin/env python
import rospy
from sensor_msgs.msg import NavSatFix
import math
import time
import numpy as np
import sensor_msgs.point_cloud2 as pc2

class GPSPublisher:
    def __init__(self):
        rospy.init_node('gps_publisher_node', anonymous=True)
        self.gps_pub = rospy.Publisher('/gps', NavSatFix, queue_size=10)
        self.path_pub = rospy.Publisher('/gps_path', pc2.PointCloud2, queue_size=10)
        
        self.gps_sub = rospy.Subscriber('/gps', NavSatFix, self.gpsCallback)
        self.slam_path_sub = rospy.Subscriber('/pcmf_anchor_pivot_node/transformed_path', pc2.PointCloud2, self.slamPathCallback)
        
        self.rate = rospy.Rate(1) # 1 Hz
            
        self.base_lat = 63.4393000
        # self.base_lat = 0.0
        self.base_lon = 10.3990500
        # self.base_lon = 0.0
        self.angle = 0
        self.path = []
        self.slam_path = []
        self.remap_lat = 63.4393000
        self.remap_lon = 10.3990500


    def gpsCallback(self, msg):
        # Convert lat/lon to local coordinates with base (63.4393, 10.39905) as (0,0)
        local_x = (msg.latitude - self.remap_lat) * 111319.9  # Approximation for conversion to meters
        local_y = (msg.longitude - self.remap_lon) * 111319.9 * math.cos(math.radians(msg.latitude))

        # Append the converted local coordinates as a tuple to the path list
        self.path.append((local_x, local_y, 0))  # Assuming a flat surface for Z=0

        # Prepare the header for the PointCloud2 message
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"

        # Convert the path list into a PointCloud2 message and publish
        cloud = pc2.create_cloud_xyz32(header, self.path)
        # self.path_pub.publish(cloud)
        
    def slamPathCallback(self, msg):
        self.slam_path = list(pc2.read_points(msg, field_names = ("x", "y", "z"), skip_nans=True))

        # shift points by 1m in x and y
        self.slam_path = [(x+10, y+10, z) for x, y, z in self.slam_path]
        self.path = self.slam_path
        self.path_pub.publish(msg)
        rospy.loginfo("Published GPS Path")
        self.rate.sleep()

    def publish_gps_data(self):
        while not rospy.is_shutdown():
            msg = NavSatFix()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "map"
            msg.status.status = 0 # status - 0: Fix, 1: Satellite, -1: No Fix
            msg.status.service = 1 # service - 1: GPS

            # Simulate movement in a circle around the base coordinates
            msg.latitude = self.base_lat + 0.0003 * math.sin(self.angle)
            msg.longitude = self.base_lon + 0.0003 * math.cos(self.angle)
            self.angle += 0.1

            self.gps_pub.publish(msg)
            rospy.loginfo("Published GPS Data: Latitude %f, Longitude %f", msg.latitude, msg.longitude)
            self.rate.sleep()

if __name__ == '__main__':
    try:
        gps_publisher = GPSPublisher()
        gps_publisher.publish_gps_data()
    except rospy.ROSInterruptException:
        pass