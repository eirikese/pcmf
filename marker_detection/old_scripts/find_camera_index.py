import cv2

def find_camera_index():
    index = 0
    # test up to 10 camera indexes
    while index < 10:
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if cap.isOpened():
            print(f"Camera index {index} is open.")
            cap.release()
        else:
            print(f"Camera index {index} is not open.")
        index += 1

find_camera_index()
