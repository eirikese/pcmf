import cv2
import apriltag
import numpy as np

# Load the camera calibration data
calibration_data = np.load('camera_calibration_live.npz')
camera_matrix = calibration_data['mtx']
dist_coeffs = calibration_data['dist']

# Prompt camera calibration data success or failure
if camera_matrix is None or dist_coeffs is None:
    print("Error: Could not load camera calibration data.")
    exit(1)
else:
    print("Camera calibration data loaded successfully from camera_calibration_live.npz")

TAG_SIZE = 0.153  # Actual size of the tag in meters
FONT_SCALE = 1  # Font scale for text annotations

def estimate_distance_to_tag(pixel_width):
    """Estimate distance to the marker based on its pixel width."""
    FOCAL_LENGTH = camera_matrix[0, 0]  # Use the focal length from the calibration data
    return (TAG_SIZE * FOCAL_LENGTH) / pixel_width

def triangulate_position(p1, p2, d1, d2):
    """Simple 2D triangulation based on distances to two markers."""
    x = (d1**2 - d2**2 + p2[0]**2 - p1[0]**2) / (2 * (p2[0] - p1[0]))
    y = np.sqrt(d1**2 - x**2) if d1**2 - x**2 > 0 else 0
    return (x, y)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    options = apriltag.DetectorOptions(families='tag36h11')
    detector = apriltag.Detector(options)

    distances = {}
    marker_centers = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Undistort the frame using calibration data
            frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, camera_matrix)

            gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)
            results = detector.detect(gray)

            for r in results:
                corners = np.int32(r.corners)
                cv2.polylines(frame_undistorted, [corners], isClosed=True, color=(0, 255, 0), thickness=3)
                pixel_width = np.linalg.norm(corners[0] - corners[1])
                distance = estimate_distance_to_tag(pixel_width)
                distances[r.tag_id] = distance
                center = np.mean(corners, axis=0)
                marker_centers[r.tag_id] = center
                cv2.putText(frame_undistorted, f"ID {r.tag_id}: {distance:.2f}m", (int(center[0]), int(center[1])-10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), 2)

            if 3 in marker_centers and 5 in marker_centers:
                # Calculate the midpoint (estimated origin) between markers 3 and 5
                midpoint = np.mean(np.array([marker_centers[3], marker_centers[5]]), axis=0).astype(int)
                # Assuming an average distance to markers for the radius, adjust as necessary
                radius = int(np.mean([distances.get(3, 0), distances.get(5, 0)]) * 5)  # Example scaling factor
                # Draw a circle around the estimated origin
                cv2.circle(frame_undistorted, (midpoint[0], midpoint[1]), radius, (0, 255, 0), 4)

            cv2.imshow('Frame', frame_undistorted)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
