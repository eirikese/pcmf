#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseArray, TransformStamped
import tf.transformations as transformations
from tf2_ros import StaticTransformBroadcaster

rospy.init_node('camera_to_marker_transform_2d')

static_broadcaster = StaticTransformBroadcaster()

fixed_marker_positions = {
    0: np.array([29.75, -169.4]),  # Marker 0
    1: np.array([30, -170]),       # Marker 1
    2: np.array([30.25, -170.6])   # Marker 2
}

def average_rotation(quaternions):
    """
    Averages the quaternions using normalized weights and returns the average quaternion.
    """
    if len(quaternions) == 1:
        return quaternions[0]

    # Convert quaternions to eulers and only consider the z component (yaw)
    z_angles = [transformations.euler_from_quaternion(q)[2] for q in quaternions]
    # Average the angles
    mean_angle = np.mean(z_angles)
    # Convert back to quaternion
    return transformations.quaternion_from_euler(0, 0, mean_angle)

def pose_callback(pose_array):
    translations = []
    quaternions = []

    for pose in pose_array.poses:
        # Extract only the xy components of the translation and the quaternion
        translation = np.array([pose.position.x, pose.position.y])
        quaternion = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])

        translations.append(translation)
        quaternions.append(quaternion)

    if translations:
        # Compute average translation
        avg_translation = np.mean(translations, axis=0)
        # Compute average rotation
        avg_quaternion = average_rotation(quaternions)
        publish_transform(avg_translation, avg_quaternion)

def publish_transform(translation, quaternion):

    # translate everything with fixed_marker_positions
    translation += fixed_marker_positions[1]
    transform = TransformStamped()
    transform.header.stamp = rospy.Time.now()
    transform.header.frame_id = "map"
    transform.child_frame_id = "camera"
    transform.transform.translation.x = translation[0]
    transform.transform.translation.y = translation[1]
    transform.transform.translation.z = 0  # No vertical movement
    transform.transform.rotation.x = quaternion[0]
    transform.transform.rotation.y = quaternion[1]
    transform.transform.rotation.z = quaternion[2]
    transform.transform.rotation.w = quaternion[3]

    static_broadcaster.sendTransform(transform)

rospy.Subscriber('/tag_poses', PoseArray, pose_callback)
rospy.spin()
