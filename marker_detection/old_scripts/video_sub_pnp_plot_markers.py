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

# Thread-safe queue for position data
position_queue = queue.Queue()

# Color and position tracking
marker_positions = {}  # Dictionary to store positions for each marker ID
marker_colors = {}  # Dictionary to store colors for each marker ID

def image_callback(img_msg):
    try:
        frame = bridge.imgmsg_to_cv2(img_msg, "bgr8")
    except cv_bridge.CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)

    current_positions = {}
    for detection in detections:
        corners = np.int32(detection.corners)
        obj_points = np.array([[0, 0, 0], [0.1, 0, 0], [0.1, 0.1, 0], [0, 0.1, 0]])
        retval, rvec, tvec = cv2.solvePnP(obj_points, corners.astype(np.float32), camera_matrix, dist_coeffs)
        if retval:
            tag_id = detection.tag_id
            tvec_flat = tvec.flatten()[:2]
            if tag_id not in marker_colors:
                # Assign a random color if new marker
                marker_colors[tag_id] = (random.random(), random.random(), random.random())
            if tag_id not in marker_positions:
                marker_positions[tag_id] = []
            marker_positions[tag_id].append(tvec_flat)
            current_positions[tag_id] = tvec_flat

    position_queue.put(current_positions)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        rospy.signal_shutdown("User requested exit.")

image_sub = rospy.Subscriber('/video_frames', Image, image_callback)

def plot_positions():
    plt.ion()
    fig, ax = plt.subplots()
    ax.grid(True)
    while not rospy.is_shutdown():
        try:
            positions = position_queue.get_nowait()
        except queue.Empty:
            plt.pause(0.1)
            continue
        
        ax.clear()
        
        for tag_id, pos in positions.items():
            color = marker_colors[tag_id]
            trace = marker_positions[tag_id]
            # Extract x and y coordinates
            x, y = zip(*trace)
            ax.plot(x, y, color=color, marker='o', linestyle='-')

        fig.canvas.draw()

    plt.close()

def main():
    plot_thread = threading.Thread(target=plot_positions)
    plot_thread.start()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
