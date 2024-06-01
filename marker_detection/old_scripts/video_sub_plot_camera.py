import cv2
import apriltag
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import threading
import queue
import random

# Initialize ROS
rospy.init_node('apriltag_detector', anonymous=True)
bridge = CvBridge()

# Initialize the apriltag detector
options = apriltag.DetectorOptions(families='tag36h11')
detector = apriltag.Detector(options)

# Camera intrinsic parameters (adjust these to your camera)
camera_matrix = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])
dist_coeffs = np.zeros(4)  # Assuming no lens distortion

# Thread-safe queue for camera positions
camera_position_queue = queue.Queue()

def image_callback(img_msg):
    try:
        frame = bridge.imgmsg_to_cv2(img_msg, "bgr8")
    except cv_bridge.CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)

    for detection in detections:
        if detection.tag_id == 1:  # Assuming Marker 1 is the origin marker
            corners = np.int32(detection.corners)
            obj_points = np.array([[0, 0, 0], [0.1, 0, 0], [0.1, 0.1, 0], [0, 0.1, 0]])  # Known marker positions in meters
            retval, rvec, tvec = cv2.solvePnP(obj_points, corners.astype(np.float32), camera_matrix, dist_coeffs)
            if retval:
                # Camera position is the negative of the translation vector (inverting the perspective)
                camera_position = -tvec.flatten()[:2]
                camera_position_queue.put(camera_position)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        rospy.signal_shutdown("User requested exit.")

image_sub = rospy.Subscriber('/video_frames', Image, image_callback)

def plot_camera_positions():
    plt.ion()
    fig, ax = plt.subplots()
    ax.grid(True)
    camera_trace = []
    while not rospy.is_shutdown():
        try:
            camera_position = camera_position_queue.get_nowait()
            camera_trace.append(camera_position)
        except queue.Empty:
            plt.pause(0.1)
            continue
        
        ax.clear()
        x, y = zip(*camera_trace)
        ax.plot(x, y, color='red', marker='o', linestyle='-')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        fig.canvas.draw()

    plt.close()

def main():
    plot_thread = threading.Thread(target=plot_camera_positions)
    plot_thread.start()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv_bridge.CvBridgeError.destroyAllWindows()

if __name__ == '__main__':
    main()
