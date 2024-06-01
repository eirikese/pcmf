import cv2
import apriltag
import numpy as np

# Load the camera calibration data
# calibration_data = np.load('camera_calibration_live.npz')
calibration_data = np.load('builtin_camera_calibration_live.npz')
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
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(4, cv2.CAP_V4L2)  # for Linux

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

            # Calculate and display camera position in 2D space
            if marker_detected[3] and marker_detected[5]:
                # X position estimation remains the same
                camera_x_position = (distances[5] - distances[3]) / 2
                
                # Y position is crudely inferred from distances; this is an oversimplification
                camera_y_position = np.sqrt(abs(distances[3]**2 - camera_x_position**2))

                cam_pos_text = f"Cam Pos: X={camera_x_position:.2f}m, Y={camera_y_position:.2f}m"
                cv2.putText(frame_undistorted, cam_pos_text, (10, frame_undistorted.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 0), 2)

            frame_undistorted = cv2.resize(frame_undistorted, (0, 0), fx=2, fy=2)
            cv2.imshow('Frame', frame_undistorted)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
