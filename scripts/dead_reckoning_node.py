#!/usr/bin/env python

import rospy
import tf
from sensor_msgs.msg import Imu, NavSatFix, PointCloud2
import sensor_msgs.point_cloud2 as pc2
import math

class DeadReckoning:
    def __init__(self):
        rospy.init_node('dead_reckoning_node')

        # Subscriber to the Ouster IMU data
        self.imu_sub = rospy.Subscriber("ouster/imu", Imu, self.imu_callback)

        # Subscriber to GPS positions for initial heading
        self.gps_sub = rospy.Subscriber("gps/fix", NavSatFix, self.gps_callback)

        # Publisher for PointCloud2 data
        self.pc_pub = rospy.Publisher("dead_reckoning_path", PointCloud2, queue_size=10)

        # Position and heading
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # GPS data for heading calculation
        self.gps_positions = []

        # Flags for initialization
        self.first_imu_received = False
        self.initial_heading_set = False

        # Initial position for reference
        self.initial_x = 0.0
        self.initial_y = 0.0

        # Timestamp for delta time calculations
        self.last_time = rospy.Time.now()

        # Store points for path visualization
        self.points = []

    def imu_callback(self, msg):
        if not self.first_imu_received:
            self.last_time = msg.header.stamp
            self.first_imu_received = True
            return

        current_time = msg.header.stamp
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time

        # Update heading based on angular velocity (yaw rotation)
        self.theta += msg.angular_velocity.z * dt

        # Calculate velocities
        vx = msg.linear_acceleration.x * dt
        vy = msg.linear_acceleration.y * dt

        # Update position
        self.x += vx * math.cos(self.theta) - vy * math.sin(self.theta)
        self.y += vx * math.sin(self.theta) + vy * math.cos(self.theta)

        # Store the relative position
        rel_x = self.x - self.initial_x
        rel_y = self.y - self.initial_y
        self.points.append([rel_x, rel_y, 0])

        # Publish the updated point cloud
        self.publish_point_cloud()

    def gps_callback(self, msg):
        # Collect initial GPS positions to calculate heading
        if len(self.gps_positions) < 2:
            self.gps_positions.append((msg.latitude, msg.longitude))
            if len(self.gps_positions) == 2:
                # Calculate initial heading from first two positions
                lat1, lon1 = self.gps_positions[0]
                lat2, lon2 = self.gps_positions[1]
                dy = lat2 - lat1
                dx = lon2 - lon1
                self.theta = math.atan2(dy, dx)
                self.initial_heading_set = True
                rospy.loginfo("Initial heading calculated from GPS.")
                # Unsubscribe to prevent further heading updates
                self.gps_sub.unregister()

    def publish_point_cloud(self):
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        cloud_points = [(p[0], p[1], p[2]) for p in self.points]
        point_cloud = pc2.create_cloud_xyz32(header, cloud_points)
        self.pc_pub.publish(point_cloud)
        print("last point x,y: ", self.x, self.y)

if __name__ == '__main__':
    dr = DeadReckoning()
    rospy.spin()
