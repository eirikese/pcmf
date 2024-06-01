#!/usr/bin/env python

import rospy
import cv2
import apriltag
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseArray, Pose
from tf.transformations import quaternion_from_euler

rospy.init_node('apriltag_detector_node')

# Initialize the CvBridge class
bridge = CvBridge()

# Initialize the OpenCV window
# cv2.startWindowThread()

# Initialize the AprilTag detector
options = apriltag.DetectorOptions(families='tag36h11')
detector = apriltag.Detector(options)

# Publisher for detected tags' poses
pose_publisher = rospy.Publisher('/tag_poses', PoseArray, queue_size=10)

# Camera parameters (adjust these to your camera)
camera_matrix = np.array([
    [800, 0, 320],       # fx, 0, cx
    [0, 800, 240],       # 0, fy, cy
    [0, 0, 1]            # 0, 0, 1
])

# Distortion coefficients
dist_coeffs = np.array([0, 0, 0, 0, 0])  # Assuming no lens distortion

def image_callback(img_msg):
    try:
        # Convert the ROS Image message to a CV2 image
        cv_image = bridge.imgmsg_to_cv2(img_msg, "bgr8")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
        return

    # Convert to grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags in the image
    detections = detector.detect(gray)

    # This will contain the poses of all detected tags
    pose_array = PoseArray()
    pose_array.header.frame_id = "camera"
    pose_array.header.stamp = rospy.Time.now()

    # Process each detection
    for detection in detections:
        # Estimate tag pose
        tag_size = 0.297  # Tag size in meters
        corners = detection.corners
        retval, rvec, tvec = cv2.solvePnP(np.array([[0, 0, 0],
                                                    [tag_size, 0, 0],
                                                    [tag_size, tag_size, 0],
                                                    [0, tag_size, 0]]),
                                          corners.astype(np.float32),
                                          camera_matrix,
                                          dist_coeffs)
        if retval:
            # Convert rotation vector to quaternion
            quat = quaternion_from_euler(rvec[0], rvec[1], rvec[2])
            
            # Create a Pose object
            pose = Pose()
            pose.position.x = tvec[0]
            pose.position.y = tvec[1]
            pose.position.z = tvec[2]
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]
            
            # Add the pose to the PoseArray
            pose_array.poses.append(pose)

    # Publish all detected poses
    pose_publisher.publish(pose_array)

    # Optionally display the image with detections
    for detection in detections:
        cv2.polylines(cv_image, [np.int32(detection.corners)], True, (0, 255, 0), 2)
    # cv2.imshow("AprilTag Detection", cv_image)
    cv2.waitKey(3)

def main():
    rospy.Subscriber('/video_frames', Image, image_callback)
    rospy.spin()

if __name__ == '__main__':
    main()
