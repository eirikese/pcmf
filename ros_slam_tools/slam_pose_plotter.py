#!/usr/bin/env python

import rospy
import matplotlib.pyplot as plt
import signal
import sys
from geometry_msgs.msg import TransformStamped
from tf.transformations import euler_from_quaternion
import numpy as np
from nav_msgs.msg import Odometry

# Global list to store position and orientation data
positions = []
orientations = []

def signal_handler(sig, frame):
    """
    Handle the Ctrl+C signal to gracefully shut down the node and plot the data.
    """
    print("\nShutting down node and plotting data...")
    save_data()
    plot_data()
    sys.exit(0)

def plot_data():
    """
    Plot the position and orientation data.
    """
    plt.figure(figsize=(8, 8))
    for pos, ori in zip(positions, orientations):
        plt.plot(pos[0], pos[1], 'bo')  # Position as blue dot
        plt.arrow(pos[0], pos[1], 0.1 * ori[0], 0.1 * ori[1], head_width=0.05, head_length=0.1, fc='r', ec='r')

    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Robot Position and Orientation (XY Birdview)')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def save_data(): # save data to npy file
    np.save('/home/mr_fusion/lidar_ws/src/pcmf/maps/mr_2710_complete_positions.npy', positions)
    np.save('/home/mr_fusion/lidar_ws/src/pcmf/maps/mr_2710_complete_orientations.npy', orientations)

def callback(data: Odometry):
    """
    Callback function for the subscribed topic.
    """
    # Extract position and orientation from Odometry message
    pos = data.pose.pose.position
    positions.append([pos.x, pos.y, pos.z])

    # Extract orientation (quaternion), and convert to Euler angles
    ori = data.pose.pose.orientation
    euler = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
    orientations.append([np.cos(euler[2]), np.sin(euler[2])])  # Store orientation as unit vector

def listener():
    """
    Initialize the ROS node and subscribe to the topic.
    """
    rospy.init_node('odom_listener', anonymous=True)
    rospy.Subscriber("/odom", Odometry, callback)
    rospy.spin()

if __name__ == '__main__':
    print('\nPose plotter initialized ...')
    signal.signal(signal.SIGINT, signal_handler)
    listener()
