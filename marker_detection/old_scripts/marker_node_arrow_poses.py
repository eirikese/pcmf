import cv2
import apriltag
import numpy as np
import rospy
from visualization_msgs.msg import Marker
import tf.transformations

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

# Initialize ROS node
rospy.init_node('marker_publisher', anonymous=True)
marker_pub = rospy.Publisher('/visualization_markers', Marker, queue_size=10)

cap = cv2.VideoCapture(4, cv2.CAP_V4L2)  # Adjust device index as necessary

if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect apriltags in the image
    detections = detector.detect(gray)

    # Keep track of detected marker IDs
    detected_marker_ids = []

    for detection in detections:
        if detection.tag_id in [1, 3, 5, 7]:
            # Extract the image coordinates of the apriltag corners
            imagePoints = np.array(detection.corners, dtype=np.float32)

            # Pose estimation
            success, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs)

            if success:
                # Draw the detected tag and its ID
                cv2.polylines(frame, [imagePoints.astype(np.int32).reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.putText(frame, f"ID {detection.tag_id}", (int(imagePoints[0][0]), int(imagePoints[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Publish marker position
                marker = Marker()
                marker.header.frame_id = "map"  # Adjust frame_id as necessary
                marker.header.stamp = rospy.Time.now()
                marker.ns = "markers"
                marker.id = detection.tag_id
                marker.type = Marker.ARROW
                marker.action = Marker.ADD
                marker.pose.position.x = tvec[0]
                marker.pose.position.y = tvec[1]
                marker.pose.position.z = tvec[2]

                # Convert rotation vector to quaternion
                R, _ = cv2.Rodrigues(rvec)
                # Adjust for coordinate system differences between OpenCV and ROS
                # This rotation aligns the OpenCV camera coordinate system with the ROS world coordinate system
                R_adjust = np.array([[0, 0, 1],
                                    [-1, 0, 0],
                                    [0, -1, 0]])
                R = np.dot(R, R_adjust)
                quaternion = tf.transformations.quaternion_from_matrix(np.vstack((np.hstack((R, [[0], [0], [0]])), [0, 0, 0, 1])))
                marker.pose.orientation.x = quaternion[0]
                marker.pose.orientation.y = quaternion[1]
                marker.pose.orientation.z = quaternion[2]
                marker.pose.orientation.w = quaternion[3]

                marker.scale.x = 0.2  # Length of the arrow
                marker.scale.y = 0.05  # Width of the arrow's shaft
                marker.scale.z = 0.05  # Height of the arrow's shaft

                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0  # Green when detected
                marker.color.a = 1.0

                marker.lifetime = rospy.Duration()  # Persistent marker
                marker_pub.publish(marker)

                detected_marker_ids.append(detection.tag_id)

    # Publish lost markers
    for lost_marker_id in [1, 3, 5, 7]:
        if lost_marker_id not in detected_marker_ids:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "markers"
            marker.id = lost_marker_id
            marker.type = Marker.ARROW
            marker.action = Marker.DELETE
            marker_pub.publish(marker)

    # Display the frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
