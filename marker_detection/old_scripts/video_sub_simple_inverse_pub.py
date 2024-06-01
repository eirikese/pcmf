import cv2
import apriltag
import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2

# Initialize ROS
rospy.init_node('apriltag_detector', anonymous=True)
bridge = CvBridge()

# Initialize the apriltag detector
options = apriltag.DetectorOptions(families='tag36h11',
                                   border=1, nthreads=4, quad_decimate=1.0, quad_blur=0.1,
                                   refine_edges=True, refine_decode=False, refine_pose=True,
                                   debug=False, quad_contours=True)
detector = apriltag.Detector(options)

# Camera intrinsic parameters for a specific camera
camera_matrix = np.array([[1069.38, 0, 360],
                          [0, 1069.38, 640],
                          [0, 0, 1]])

# Distortion coefficients, assuming no distortion for simplification
dist_coeffs = np.array([0, 0, 0, 0, 0])

# ROS Publisher for PointCloud2
camera_trace_pub = rospy.Publisher('/camera_trace', PointCloud2, queue_size=10)

# Define fixed marker position for marker 1
fixed_marker_position = np.array([30, -170])  # Marker 1

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
        if detection.tag_id == 1:
            # Marker size in meters
            tag_size = 0.297
            # Extract camera parameters
            fx = camera_matrix[0, 0]
            fy = camera_matrix[1, 1]
            cx = camera_matrix[0, 2]
            cy = camera_matrix[1, 2]
            # Correct camera_params format
            camera_params = (fx, fy, cx, cy)

            pose, e0, e1 = detector.detection_pose(detection, camera_params, tag_size)
            if pose is not None:
                # Invert the pose matrix to get the camera's position and orientation relative to the marker
                pose_inv = np.linalg.inv(pose)
                camera_position = pose_inv[:3, 3]  # Extract the translation components
                camera_positions.append(camera_position.tolist())

    # Publish camera positions as a PointCloud2 message
    if camera_positions:
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1)]
        cloud = pc2.create_cloud(header, fields, camera_positions)
        camera_trace_pub.publish(cloud)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        rospy.signal_shutdown("User requested exit.")


image_sub = rospy.Subscriber('/video_frames', Image, image_callback)

def main():
    rospy.spin()

if __name__ == '__main__':
    main()
