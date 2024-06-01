import cv2
import apriltag
import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import matplotlib.pyplot as plt

# Initialize ROS
rospy.init_node('apriltag_detector', anonymous=True)
bridge = CvBridge()

# Initialize the apriltag detector
options = apriltag.DetectorOptions(families='tag36h11')
detector = apriltag.Detector(options)

# Camera intrinsic parameters
# camera_matrix = np.array([
#     [800, 0, 360],
#     [0, 1193.47, 640],
#     [0, 0, 1]
# ])

camera_matrix = np.array([
    [800, 0, 360],       # Adjusted focal length on x-axis
    [1193.47, 0, 640],   # Adjusted focal length on y-axis
    [0, 0, 1]            # Affine components remain standard
])

# Distortion coefficients for a slight wide-angle lens
dist_coeffs = np.array([-0.15, 0.02, 0, 0, 0])

# ROS Publisher for PointCloud2
camera_trace_pub = rospy.Publisher('/camera_trace', PointCloud2, queue_size=10)

# Local to Map frame transformation (Define these appropriately)
# Rotate 150 degrees clockwise
R = np.eye(2)  # Identity matrix placeholder
R = np.array([[np.cos(np.pi * -150 / 180), -np.sin(np.pi * -150 / 180)],
              [np.sin(np.pi * -150 / 180), np.cos(np.pi * -150 / 180)]])
t = np.array([30, -170])  # Translation vector placeholder

# plot axis before and after transformation in plt
# use xy = red green convention
plt.quiver(0, 0, 1, 0, color='red', scale=0.1)
plt.quiver(0, 0, 0, 1, color='green', scale=0.1)
plt.quiver(t[0], t[1], R[0, 0], R[1, 0], color='red', scale=1)
plt.quiver(t[0], t[1], R[0, 1], R[1, 1], color='green', scale=1)
plt.axis('equal')
plt.show()

# Define fixed marker positions in the local frame
fixed_marker_positions_local = {
    0: np.array([-1, 0]),  # Marker 0
    1: np.array([0, 0]),   # Marker 1
    2: np.array([1, 0])    # Marker 2
}

def image_callback(img_msg):
    try:
        frame = bridge.imgmsg_to_cv2(img_msg, "bgr8")
    except cv_bridge.CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)
    camera_positions_map_frame = []

    for detection in detections:
        if detection.tag_id in fixed_marker_positions_local:
            corners = np.int32(detection.corners)
            marker_size = 0.297
            obj_points = np.array([
                [0, 0, 0],
                [marker_size, 0, 0],
                [marker_size, marker_size, 0],
                [0, marker_size, 0]
            ])
            retval, rvec, tvec = cv2.solvePnP(obj_points, corners.astype(np.float32), camera_matrix, dist_coeffs)
            if retval:
                # Local frame calculations
                camera_position_local = tvec.flatten()[:2] + fixed_marker_positions_local[detection.tag_id]
                # Transform to map frame
                camera_position_map = R @ camera_position_local + t
                camera_positions_map_frame.append(list(camera_position_map) + [0.0])  # z-coordinate is 0

    if camera_positions_map_frame:
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1)]
        cloud = pc2.create_cloud(header, fields, camera_positions_map_frame)
        camera_trace_pub.publish(cloud)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        rospy.signal_shutdown("User requested exit.")

image_sub = rospy.Subscriber('/video_frames', Image, image_callback)

def main():
    rospy.spin()

if __name__ == '__main__':
    main()
