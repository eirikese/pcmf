import cv2
import apriltag
import numpy as np
import rospy
from visualization_msgs.msg import Marker

# Load the camera calibration data
calibration_data = np.load('builtin_camera_calibration_live.npz')
camera_matrix = calibration_data['mtx']
dist_coeffs = calibration_data['dist']

print("Camera calibration data loaded successfully.")

TAG_SIZE = 0.153  # Size of the tag in meters
FONT_SCALE = 0.7  # Font scale for text annotations

def estimate_distance_to_tag(pixel_width):
    """Estimate the distance to the marker based on its pixel width."""
    FOCAL_LENGTH = camera_matrix[0, 0]  # Use the focal length from the calibration data
    return (TAG_SIZE * FOCAL_LENGTH) / pixel_width

def circle_intersection(c1, r1, c2, r2):
    """Calculate intersection points of two circles."""
    d = np.linalg.norm(np.array(c1) - np.array(c2))
    if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
        return []  # No intersection or one circle within the other

    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = np.sqrt(max(r1**2 - a**2, 0))
    x2 = c1[0] + a * (c2[0] - c1[0]) / d
    y2 = c1[1] + a * (c2[1] - c1[1]) / d
    x3 = x2 + h * (c2[1] - c1[1]) / d
    y3 = y2 - h * (c2[0] - c1[0]) / d

    x4 = x2 - h * (c2[1] - c1[1]) / d
    y4 = y2 + h * (c2[0] - c1[0]) / d

    return [(x3, y3), (x4, y4)]

def estimate_camera_position(marker_positions, distances):
    """Estimate the camera position using the positions and distances of three markers."""
    valid_intersections = []
    marker_ids = [id for id in marker_positions.keys() if id in distances]

    # Ensure there are at least two markers with distances measured
    if len(marker_ids) < 2:
        return None

    for i in range(len(marker_ids)):
        for j in range(i + 1, len(marker_ids)):
            id1, id2 = marker_ids[i], marker_ids[j]
            p1, p2 = marker_positions[id1], marker_positions[id2]
            d1, d2 = distances[id1], distances[id2]

            # Find intersections between circles centered at p1 and p2 with radii d1 and d2
            inters = circle_intersection(p1, d1, p2, d2)
            for inter in inters:
                if inter[1] > 0:  # Filter for positive y-values
                    valid_intersections.append(inter)

    if not valid_intersections:
        return None  # No valid intersections found

    # Average the valid intersections to estimate the camera position
    avg_x = sum(pt[0] for pt in valid_intersections) / len(valid_intersections)
    avg_y = sum(pt[1] for pt in valid_intersections) / len(valid_intersections)

    return (avg_x, avg_y)


def create_marker(marker_id, position, color, orientation=(0, 0, 0, 1)):
    """Create a Marker message."""
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "marker_positions"
    marker.id = marker_id
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = position[0]
    marker.pose.position.y = position[1]
    marker.pose.position.z = 0  # Assuming markers are in the XY plane
    marker.pose.orientation.x = orientation[0]
    marker.pose.orientation.y = orientation[1]
    marker.pose.orientation.z = orientation[2]
    marker.pose.orientation.w = orientation[3]
    marker.scale.x = 0.3  # Adjust size as needed
    marker.scale.y = 0.3
    marker.scale.z = 0.3
    marker.color.a = 1.0
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    return marker

def main():
    rospy.init_node('marker_positions_publisher', anonymous=True)
    pub_markers = rospy.Publisher('/visualization_markers', Marker, queue_size=10)

    cap = cv2.VideoCapture(4, cv2.CAP_V4L2)  # Adjust device index as necessary

    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    options = apriltag.DetectorOptions(families='tag36h11')
    detector = apriltag.Detector(options)

    # Initial marker positions (ID 1 at (0,0), ID 3 at (-1,0), ID 5 at (1,0))
    marker_positions = {1: (0, 0), 3: (-1, 0), 5: (1, 0)}

    try:
        while not rospy.is_shutdown():
            ret, frame = cap.read()
            if not ret:
                break

            frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, camera_matrix)
            gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)
            results = detector.detect(gray)

            distances = {}
            marker_detected = {1: False, 3: False, 5: False}  # Update to include the third marker

            for r in results:
                if r.tag_id in marker_positions:
                    corners = np.int32(r.corners)
                    cv2.polylines(frame_undistorted, [corners], isClosed=True, color=(0, 255, 0), thickness=3)
                    pixel_width = np.linalg.norm(corners[0] - corners[1])
                    distance = estimate_distance_to_tag(pixel_width)
                    distances[r.tag_id] = distance
                    marker_detected[r.tag_id] = True

                    # Display detection status on frame
                    cv2.putText(frame_undistorted, f"ID {r.tag_id}: {distance:.2f}m Detected", (int(corners[0][0]), int(corners[0][1])-10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), 2)

                    # Publish marker positions
                    color = (0, 1, 0) if marker_detected[r.tag_id] else (1, 0, 0)
                    marker = create_marker(r.tag_id, marker_positions[r.tag_id], color)
                    pub_markers.publish(marker)
            
            # if marker not detected, publish marker with red color
            for id in marker_detected:
                if not marker_detected[id]:
                    marker = create_marker(id, marker_positions[id], (1, 0, 0))
                    pub_markers.publish(marker)

            # Estimate and publish camera position if at least three markers are detected
            if sum(marker_detected.values()) >= 3:
                camera_position = estimate_camera_position(marker_positions, distances)
                if camera_position:
                    camera_marker = create_marker(99, camera_position, (0, 0, 1))  # Blue for the camera position
                    pub_markers.publish(camera_marker)
            else: # if camera position not detected, delete the marker
                camera_marker = create_marker(99, (0, 0), (0, 0, 1))
                camera_marker.action = Marker.DELETE
                pub_markers.publish(camera_marker)

            cv2.imshow('Frame', frame_undistorted)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
