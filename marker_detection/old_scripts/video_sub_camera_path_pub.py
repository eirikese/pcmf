import cv2
import apriltag
import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import threading
import queue
import matplotlib.pyplot as plt

# Initialize ROS
rospy.init_node('apriltag_detector', anonymous=True)
bridge = CvBridge()

# Initialize the apriltag detector
options = apriltag.DetectorOptions(families='tag36h11')
detector = apriltag.Detector(options)

# Camera intrinsic parameters (adjust these to your camera)
# camera_matrix = np.array([
#     [800, 0, 360],       # Adjusted focal length on x-axis
#     [1193.47, 0, 640],   # Adjusted focal length on y-axis
#     [0, 0, 1]            # Affine components remain standard
# ])

camera_matrix = np.array([
    [800, 0, 360],       # fx, 0, cx
    [0, 1100, 640],       # 0, fy, cy
    [0, 0, 1]            # 0, 0, 1
])

# Distortion coefficients for a slight wide-angle lens
dist_coeffs = np.array([-0.15, 0.02, 0, 0, 0])  # Starting with an estimation for mild barrel distortion

# ROS Publisher for PointCloud2
camera_trace_pub = rospy.Publisher('/camera_trace', PointCloud2, queue_size=10)

# Define fixed marker positions (assuming markers are defined relative to some reference point)
fixed_marker_positions = {
    0: np.array([29.75, -169.4]),  # Marker 0 one meter to the left of Marker 1
    1: np.array([30, -170]),      # Marker 1
    2: np.array([30.25, -170.6])  # Marker 2 one meter to the right of Marker 1
}

# Load map data
map_file_path = "/home/eirik/lidar_ws/src/pcmf/maps/all_points.npy" # bridge map data
# map_file_path = "/home/eirik/lidar_ws/src/pcmf/maps/bay_lines_v3.npy" # brattorkaia map data
map_data = np.load(map_file_path)

# plot marker positions with plt together with map data
plt.scatter([pos[0] for pos in fixed_marker_positions.values()], [pos[1] for pos in fixed_marker_positions.values()])
plt.plot(map_data[:, 0], map_data[:, 1], 'orange')
plt.axis('equal')
pad = 10
plt.xlim([29.5 - pad, 30.5 + pad])
plt.ylim([-171 - pad, -169 + pad])
# plt.show()

def image_callback(img_msg):
    try:
        frame = bridge.imgmsg_to_cv2(img_msg, "bgr8")
    except cv_bridge.CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)
    camera_positions = []

    for detection in detections:
        if detection.tag_id in fixed_marker_positions:
            corners = np.int32(detection.corners)
            # Define object points based on actual marker size in meters
            marker_size = 0.297  # Marker size in meters
            obj_points = np.array([
                [0, 0, 0],
                [marker_size, 0, 0],
                [marker_size, marker_size, 0],
                [0, marker_size, 0]
            ])
            retval, rvec, tvec = cv2.solvePnP(obj_points, corners.astype(np.float32), camera_matrix, dist_coeffs)
            if retval:
                # Calculate the camera's position relative to the fixed marker
                camera_position = tvec.flatten()[:2] + fixed_marker_positions[detection.tag_id]
                camera_positions.append(list(camera_position) + [0.0])  # z-coordinate is 0

    # # average and make one camera position
    # if camera_positions:
    #     camera_positions = np.array(camera_positions)
    #     avg_camera_position = np.mean(camera_positions, axis=0)
    #     camera_positions = [avg_camera_position]

    # Publish camera positions as a PointCloud2 message
    if camera_positions:
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"  # All messages use frame "map"
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1)]
        cloud = pc2.create_cloud(header, fields, camera_positions)
        camera_trace_pub.publish(cloud)

    # cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        rospy.signal_shutdown("User requested exit.")

image_sub = rospy.Subscriber('/video_frames', Image, image_callback)

def main():
    rospy.spin()

if __name__ == '__main__':
    main()
