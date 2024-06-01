import cv2
import apriltag
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Initialize ROS
rospy.init_node('apriltag_detector', anonymous=True)
bridge = CvBridge()

# Initialize the apriltag detector
options = apriltag.DetectorOptions(families='tag36h11')
detector = apriltag.Detector(options)

def image_callback(img_msg):
    try:
        # Convert ROS image to OpenCV format
        frame = bridge.imgmsg_to_cv2(img_msg, "bgr8")
    except cv_bridge.CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
        return

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags in the image
    detections = detector.detect(gray)

    # Draw detection results on the frame
    for detection in detections:
        # Draw the bounding box
        corners = np.int32(detection.corners)
        cv2.polylines(frame, [corners], isClosed=True, color=(0, 255, 0), thickness=2)

        # Put the tag ID near the first corner
        cv2.putText(frame, str(detection.tag_id), tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        rospy.signal_shutdown("User requested exit.")

# Subscribe to the video frame topic
image_sub = rospy.Subscriber('/video_frames', Image, image_callback)

def main():
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
