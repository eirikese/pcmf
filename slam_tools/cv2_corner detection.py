import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the .npy file
file_path = "/home/mr_fusion/lidar_ws/src/pcmf/maps/reference_map_mr_partial.npy"
data = np.load(file_path)

# Check the shape of the loaded data
# Convert the coordinates to a photo with a given pixel_to_meter_ratio
pixel_to_meter_ratio = 100
x_pixels = (data[:, 0] - data[:, 0].min()) * pixel_to_meter_ratio
y_pixels = (data[:, 1] - data[:, 1].min()) * pixel_to_meter_ratio

# Create a blank image
width = int(x_pixels.max()) + 1
height = int(y_pixels.max()) + 1
image = np.zeros((height, width), dtype=np.uint8)

# Draw the points on the image
for x, y in zip(x_pixels.astype(int), y_pixels.astype(int)):
    image[y, x] = 255

# Show the image
# plt.figure(figsize=(10, 6))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.imshow(image, cmap='gray')
# plt.xlabel('X coordinate (pixels)')
# plt.ylabel('Y coordinate (pixels)')
# plt.title('Original Data as image')
# plt.show()

# Detect corners using OpenCV
corners = cv2.goodFeaturesToTrack(image, maxCorners=10, qualityLevel=0.01, minDistance=10, useHarrisDetector=True)
if corners is not None:
    corners = np.int0(corners)

    # Convert corner positions back to meters
    corner_positions_meters = corners.reshape(-1, 2) / pixel_to_meter_ratio
    corner_positions_meters[:, 0] += data[:, 0].min()
    corner_positions_meters[:, 1] += data[:, 1].min()

    # Plot the original npy data and the detected corners
    plt.figure(figsize=(10, 6))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot(data[:, 0], data[:, 1], 'ro', label='Original Data')
    plt.plot(corner_positions_meters[:, 0], corner_positions_meters[:, 1], 'bx', label='Detected Corners')
    plt.legend()
    plt.xlabel('X coordinate (meters)')
    plt.ylabel('Y coordinate (meters)')
    plt.title('Original Data with Detected Corners')
    plt.show()
else:
    print("No corners were detected.")