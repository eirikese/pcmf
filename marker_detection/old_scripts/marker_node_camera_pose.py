import cv2
import apriltag
import numpy as np
import rospy
from visualization_msgs.msg import Marker
import tf.transformations as tf_trans

# Define the real-world coordinates of apriltag corners for pose estimation
TAG_SIZE = 0.153  # Meters
TAG_HALF = TAG_SIZE / 2
objectPoints = np.array([
    [-TAG_HALF, -TAG_HALF, 0],
    [TAG_HALF, -TAG_HALF, 0],
    [TAG_HALF, TAG_HALF, 0],
    [-TAG_HALF, TAG_HALF, 0]
], dtype=np.float32)

# Load the camera calibration data
calibration_data = np.load('builtin_camera_calibration_live.npz')
cameraMatrix = calibration_data['mtx']
distCoeffs = calibration_data['dist']

# Initialize the apriltag detector
options = apriltag.DetectorOptions(families='tag36h11')
detector = apriltag.Detector(options)

# Known marker positions
known_marker_positions = {
    1: np.array([0, 1, 0]),
    3: np.array([0, 0, 0]),
    5: np.array([0, -1, 0])
}

# Initialize ROS node
rospy.init_node('marker_publisher', anonymous=True)
marker_pub = rospy.Publisher('/visualization_markers', Marker, queue_size=10)

def convert_opencv_to_ros_translation(tvec):
    """Converts a translation vector from OpenCV to ROS coordinate system."""
    return np.array([tvec[2], -tvec[0], -tvec[1]])

def invert_pose(rvec, tvec_ros):
    R, _ = cv2.Rodrigues(rvec)
    R_inv = np.transpose(R)
    tvec_inv = -np.dot(R_inv, tvec_ros)
    rvec_inv, _ = cv2.Rodrigues(R_inv)
    return rvec_inv, tvec_inv

def average_poses(poses):
    avg_rvec = np.mean([pose[0] for pose in poses], axis=0)
    avg_tvec = np.mean([pose[1] for pose in poses], axis=0)
    return avg_rvec, avg_tvec

def publish_marker(id, position, color, is_detected=True, orientation=None):
    marker = Marker()
    marker.header.frame_id = "map"  # Use appropriate reference frame
    marker.header.stamp = rospy.Time.now()
    marker.ns = "apriltag_markers"
    marker.id = id
    marker.type = Marker.ARROW
    marker.action = Marker.ADD
    marker.pose.position.x = position[0]
    marker.pose.position.y = position[1]
    marker.pose.position.z = position[2]
    
    if orientation is None:
        orientation = tf_trans.quaternion_from_euler(0, 0, 0)
    marker.pose.orientation.x = orientation[0]
    marker.pose.orientation.y = orientation[1]
    marker.pose.orientation.z = orientation[2]
    marker.pose.orientation.w = orientation[3]
    
    marker.scale.x = 0.4
    marker.scale.y = 0.1
    marker.scale.z = 0.1
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = 1.0  # Alpha
    marker.lifetime = rospy.Duration()

    if not is_detected and id == -1:
        marker.action = Marker.DELETE
    marker_pub.publish(marker)

cap = cv2.VideoCapture(4, cv2.CAP_V4L2)  # Adjust the device index as necessary

if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)
    detected_marker_ids = []
    inverse_poses = []

    for detection in detections:
        marker_id = detection.tag_id
        if marker_id in known_marker_positions.keys():
            success, rvec, tvec = cv2.solvePnP(objectPoints, np.array(detection.corners, dtype=np.float32), cameraMatrix, distCoeffs)
            if success:
                # Convert to ROS coordinates before any calculations
                tvec_ros = convert_opencv_to_ros_translation(tvec)
                detected_marker_ids.append(marker_id)
                rvec_inv, tvec_inv = invert_pose(rvec, tvec_ros)
                inverse_poses.append((rvec_inv, tvec_inv))
                publish_marker(marker_id, known_marker_positions[marker_id], (0, 1, 0))  # Green for detected

    for marker_id, position in known_marker_positions.items():
        if marker_id not in detected_marker_ids:
            publish_marker(marker_id, position, (1, 0, 0))  # Red for not detected

    if len(inverse_poses) > 2:
        avg_rvec, avg_tvec = average_poses(inverse_poses)
        orientation = tf_trans.quaternion_from_matrix(np.vstack((np.hstack((cv2.Rodrigues(avg_rvec)[0], [[0], [0], [0]])), [0, 0, 0, 1])))
        publish_marker(-1, avg_tvec, (0, 0, 1), is_detected=True, orientation=orientation)  # Blue for camera pose
    else: # remove the blue marker if there are less than 3 markers detected
        publish_marker(-1, avg_tvec, (0, 0, 1), is_detected=False)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
