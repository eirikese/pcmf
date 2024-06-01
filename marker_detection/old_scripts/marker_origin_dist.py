import cv2
import apriltag
import numpy as np

# Load the camera calibration data
calibration_data = np.load('camera_calibration_live.npz')
camera_matrix = calibration_data['mtx']
dist_coeffs = calibration_data['dist']

print("Camera calibration data loaded successfully from camera_calibration_live.npz")

TAG_SIZE = 0.153  # Actual size of the tag in meters
FONT_SCALE = 0.7  # Font scale for text annotations

def estimate_distance_to_tag(pixel_width):
    """Estimate distance to the marker based on its pixel width."""
    FOCAL_LENGTH = camera_matrix[0, 0]  # Use the focal length from the calibration data
    return (TAG_SIZE * FOCAL_LENGTH) / pixel_width

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

            marker_detected = {3: False, 5: False}
            for r in results:
                corners = np.int32(r.corners)
                cv2.polylines(frame_undistorted, [corners], isClosed=True, color=(0, 255, 0), thickness=3)
                pixel_width = np.linalg.norm(corners[0] - corners[1])
                distance = estimate_distance_to_tag(pixel_width)
                distances[r.tag_id] = distance
                center = np.mean(corners, axis=0)
                marker_centers[r.tag_id] = center
                marker_detected[r.tag_id] = True
                cv2.putText(frame_undistorted, f"ID {r.tag_id}: {distance:.2f}m", (int(center[0]), int(center[1])-10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), 2)

            if marker_detected[3] and marker_detected[5]:
                # Assuming a simplistic scenario where camera is aligned with the x-axis
                # The distance from origin can be approximated by averaging the distances of the two markers
                distance_from_origin = np.mean([distances[3], distances[5]])
                distance_text = f"Distance from Origin: {distance_from_origin:.2f}m"
                cv2.putText(frame_undistorted, distance_text, (10, frame_undistorted.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 0), 2)

            # double window size, mirror before displaying
            frame_undistorted = cv2.resize(frame_undistorted, (0, 0), fx=2, fy=2)
            cv2.imshow('Frame', frame_undistorted)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
