#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Transform
import tf.transformations
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class TransformationHistoryPlotter:
    def __init__(self):
        rospy.init_node('ekf_transformation_plotter', anonymous=True)
        
        self.transformation_history = []
        self.EKF_transformation_history = []

        rospy.Subscriber("pcmf_anchor_pivot_node/transformation_history", Transform, self.transformation_history_callback)
        rospy.Subscriber("pcmf_anchor_pivot_node/EKF_transformation_history", Transform, self.EKF_transformation_history_callback)

        # Create the initial plot
        self.fig, self.axs = plt.subplots(3, 1, figsize=(5, 9))
        self.x_normal, self.y_normal, self.yaw_normal = [], [], []
        self.x_ekf, self.y_ekf, self.yaw_ekf = [], [], []
        # set window position to upper left
        self.fig.canvas.manager.window.wm_geometry("+0+0")

        # Set window title
        self.fig.canvas.set_window_title('EKF Transformation Plotter')
        self.sliding_window = 100
        # pad subplot to avoid overlapping
        # plt.tight_layout(pad=5.0)

        # Create animation
        self.animation = FuncAnimation(self.fig, self.update_plot, blit=False, interval=100)  # Update every 100 milliseconds

        # Register the shutdown callback
        rospy.on_shutdown(self.shutdown_callback)

    def transformation_history_callback(self, data):
        if len(self.transformation_history) >= self.sliding_window:
            self.transformation_history.pop(0)  # Remove the oldest item
        self.transformation_history.append(data)

    def EKF_transformation_history_callback(self, data):
        if len(self.EKF_transformation_history) >= self.sliding_window:
            self.EKF_transformation_history.pop(0)  # Remove the oldest item
        self.EKF_transformation_history.append(data)

    def update_plot(self, frame):
        # Clear the previous data
        for ax in self.axs:
            ax.clear()

        # Update the data
        self.x_normal = [data.translation.x for data in self.transformation_history]
        self.y_normal = [data.translation.y for data in self.transformation_history]
        self.yaw_normal = [tf.transformations.euler_from_quaternion([data.rotation.x, data.rotation.y, data.rotation.z, data.rotation.w])[2] for data in self.transformation_history]

        self.x_ekf = [data.translation.x for data in self.EKF_transformation_history]
        self.y_ekf = [data.translation.y for data in self.EKF_transformation_history]
        self.yaw_ekf = [tf.transformations.euler_from_quaternion([data.rotation.x, data.rotation.y, data.rotation.z, data.rotation.w])[2] for data in self.EKF_transformation_history]

        # use degrees for yaw
        self.yaw_normal = [yaw * 180 / 3.14159 for yaw in self.yaw_normal]
        self.yaw_ekf = [yaw * 180 / 3.14159 for yaw in self.yaw_ekf]

        # Plot the updated data
        self.axs[0].plot(self.x_normal, label='Normal Transformation X')
        self.axs[0].plot(self.x_ekf, label='EKF Transformation X', linestyle='--')
        self.axs[0].set_ylabel('North Translation [m]')
        self.axs[0].grid(True)
        self.axs[0].legend()

        self.axs[1].plot(self.y_normal, label='Normal Transformation Y')
        self.axs[1].plot(self.y_ekf, label='EKF Transformation Y', linestyle='--')
        self.axs[1].set_ylabel('East Translation [m]')
        self.axs[1].grid(True)
        self.axs[1].legend()

        self.axs[2].plot(self.yaw_normal, label='Normal Rotation Yaw')
        self.axs[2].plot(self.yaw_ekf, label='EKF Rotation Yaw', linestyle='--')
        self.axs[2].set_xlabel('Time [s]')
        self.axs[2].set_ylabel('Yaw Rotation [deg]')
        self.axs[2].grid(True)
        self.axs[2].legend()

    def shutdown_callback(self):
        # Close the plot when the node is shutdown
        plt.close()

if __name__ == '__main__':
    plotter = TransformationHistoryPlotter()
    plt.show()
    rospy.spin()
