import numpy as np
import cv2

class KalmanFilter:
    def __init__(self, dt=1, processNoiseCov=1e-2, measurementNoiseCov=1e-1, errorCovPost=1.0):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * processNoiseCov
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurementNoiseCov
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * errorCovPost

    def predict(self):
        return self.kf.predict()

    def correct(self, measurement):
        return self.kf.correct(measurement)
