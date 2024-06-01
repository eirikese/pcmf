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
import std_msgs.msg

# Initialize ROS
rospy.init_node('apriltag_detector', anonymous=True)
bridge = CvBridge()

# Initialize the apriltag detector
options = apriltag.DetectorOptions(families='tag36h11')
detector = apriltag.Detector(options)

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
# plt.scatter([pos[0] for pos in fixed_marker_positions.values()], [pos[1] for pos in fixed_marker_positions.values()])
# plt.plot(map_data[:, 0], map_data[:, 1], 'orange')
# plt.axis('equal')
# pad = 10
# plt.xlim([29.5 - pad, 30.5 + pad])
# plt.ylim([-171 - pad, -169 + pad])
# plt.show()

def image_callback(img_msg):
    try:
        frame = bridge.imgmsg_to_cv2(img_msg, "bgr8")
    except cv_bridge.CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)

    # find euclidean distances to each marker
    distances_to_markers = {}
    camera_positions = []

    for detection in detections:
        # Calculate the distance based on the focal length, actual tag size, and perceived tag size in pixels
        # tag_size = 0.297  # actual size of the tag in meters
        tag_size = 0.25  # actual size of the tag in meters
        p1, p2 = detection.corners[0], detection.corners[2]  # diagonal corners
        perceived_size_pixels = np.linalg.norm(p1 - p2)
        focal_length_pixels = (camera_matrix[0, 0] + camera_matrix[1, 1]) / 2
        distance = (tag_size * focal_length_pixels) / perceived_size_pixels
        distances_to_markers[detection.tag_id] = distance

        # Extract marker's fixed position
        marker_position = fixed_marker_positions[detection.tag_id]
        
        # Triangulate camera position based on distance and marker positions
        # Assuming a simple inverse distance weighting or similar triangulation for 2D
        weighted_position = marker_position + (distance * np.array([np.cos(np.pi/4), np.sin(np.pi/4)]))
        camera_positions.append(weighted_position)

    # Calculate average to estimate camera position if more than 2 markers are detected
    if camera_positions and len(camera_positions) > 2:
        estimated_camera_position = np.mean(camera_positions, axis=0)
        print("Estimated Camera Position: ", estimated_camera_position)

        # Publishing camera position to ROS
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        estimated_camera_position = np.append(estimated_camera_position, 0)
        point_cloud = pc2.create_cloud_xyz32(header, [estimated_camera_position.tolist()])
        camera_trace_pub.publish(point_cloud)

    # Display the frame with detections
    # cv2.imshow('Frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     rospy.signal_shutdown("User requested exit.")


image_sub = rospy.Subscriber('/video_frames', Image, image_callback)

def main():
    rospy.spin()

if __name__ == '__main__':
    main()
