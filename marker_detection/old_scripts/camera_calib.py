import numpy as np
import cv2
import time

# Define the dimensions of the chessboard corners (width and height)
CHESSBOARD_SIZE = (9, 6)

# Size of a square in your chosen unit (e.g., millimeters)
square_size = 22.0

# Termination criteria for the corner sub-pixel refinement process
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points based on the real-world dimensions of the chessboard
objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points from all the frames
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane

# Start webcam capture
cap = cv2.VideoCapture(4)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Capture parameters
capture_interval = 0.5 # seconds
last_capture_time = time.time()
min_frames = 20  # Minimum number of frames to collect before calibration

print("Auto-capturing frames for calibration...")

while len(objpoints) < min_frames:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret == True:
        cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners, ret)

    cv2.imshow('Frame', frame)

    if time.time() - last_capture_time > capture_interval and ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        print(f"Frame captured. Total frames: {len(objpoints)}")
        last_capture_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Allows early exit
        break

cv2.destroyAllWindows()
cap.release()

# Calibration
if len(objpoints) >= min_frames:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Print out the calibration results
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)
    print("Rotation Vectors:\n", rvecs)
    print("Translation Vectors:\n", tvecs)

    # Optionally, you can save the calibration results for later use
    np.savez('camera_calibration_live2.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
else:
    print("Not enough frames were collected for calibration.")
