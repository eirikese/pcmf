#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Imu, PointCloud2
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2 as pc2
import math
import tf

class DeadReckoning:
    def __init__(self):
        rospy.init_node('dead_reckoning_node', anonymous=True)
        self.rate = rospy.Rate(10)  # 10 Hz

        self.imu_sub = rospy.Subscriber("ouster/imu", Imu, self.imu_callback, queue_size=1, buff_size=1) # limit to 1 hz
        self.gps_sub = rospy.Subscriber("/gps/local_fix", Marker, self.gps_callback)
        self.gps_heading_sub = rospy.Subscriber("/gps/heading", Odometry, self.gps_heading_callback)
        self.pc_pub = rospy.Publisher("dead_reckoning_path", PointCloud2, queue_size=10)

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.first_imu_received = False
        self.initial_position_set = False
        self.initial_heading_set = False

        self.last_time = None
        self.points = []

    def imu_callback(self, msg):
        if not (self.initial_position_set and self.initial_heading_set and self.first_imu_received):
            self.last_time = msg.header.stamp
            self.first_imu_received = True
            return

        current_time = msg.header.stamp
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time

        # Update heading based on angular velocity (yaw rotation)
        self.theta += msg.angular_velocity.z * dt

        # Calculate velocities using the small-angle approximation for small dt
        vx = msg.linear_acceleration.x * dt
        vy = msg.linear_acceleration.y * dt

        # Update position using the current heading
        self.x += vx * math.cos(self.theta) - vy * math.sin(self.theta)
        self.y += vx * math.sin(self.theta) + vy * math.cos(self.theta)

        rospy.loginfo(f"Position updated to x={self.x}, y={self.y}")

        # Store the relative position
        self.points.append([self.x, self.y, 0])
        self.publish_point_cloud()

    def gps_callback(self, msg):
        if not self.initial_position_set:
            self.x = msg.pose.position.x
            self.y = msg.pose.position.y
            self.initial_position_set = True
            rospy.loginfo("Initial GPS position set.")

    def gps_heading_callback(self, msg):
        if not self.initial_heading_set:
            orientation_q = msg.pose.pose.orientation
            euler = tf.transformations.euler_from_quaternion([
                orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w
            ])
            self.theta = euler[2]  # yaw
            self.initial_heading_set = True
            rospy.loginfo("Initial GPS heading set.")

    def publish_point_cloud(self):
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        cloud_points = [(p[0], p[1], 0) for p in self.points]
        point_cloud = pc2.create_cloud_xyz32(header, cloud_points)
        self.pc_pub.publish(point_cloud)

if __name__ == '__main__':
    dr = DeadReckoning()
    rospy.spin()
