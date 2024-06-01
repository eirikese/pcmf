#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from collections import deque

class EuclideanErrorPlotter:
    def __init__(self):
        self.delays = [1, 2, 3, 4, 5]  # Delays in seconds
        rospy.init_node('euclidean_error_plotter', anonymous=True)
        
        self.transformed_path_points = []
        self.gps_locations = deque(maxlen=200)  # Adjusted maxlen for maximum delay
        self.euclidean_errors = {delay: [] for delay in self.delays}

        rospy.Subscriber("/pcmf_anchor_pivot_node/transformed_path", PointCloud2, self.transformed_path_callback)
        rospy.Subscriber("/gps/local_fix", Marker, self.local_fix_callback)

        self.fig = plt.figure(figsize=(10, 4))
        self.fig.canvas.set_window_title('GNSS / SLAM Position Euclidean Error Plotter')
        gs = gridspec.GridSpec(1, 4)
        self.ax_time = self.fig.add_subplot(gs[0, :3])
        self.ax_box = self.fig.add_subplot(gs[0, 3])

        plt.tight_layout(w_pad=3.0, h_pad=3.0, pad=3.0)
        self.fig.canvas.manager.window.wm_geometry("+600+650")

        self.animation = FuncAnimation(self.fig, self.update_plot, blit=False, interval=1000)

        rospy.on_shutdown(self.shutdown_callback)

    def transformed_path_callback(self, data):
        gen = pc2.read_points(data, skip_nans=True)
        self.transformed_path_points = [p for p in gen]

    def local_fix_callback(self, data):
        self.gps_locations.append((rospy.get_time(), (data.pose.position.x, data.pose.position.y, data.pose.position.z)))

    def get_gps_location_delayed(self, delay_time):
        target_time = rospy.get_time() - delay_time
        closest_location = None
        for timestamp, location in self.gps_locations:
            if timestamp <= target_time:
                closest_location = location
            else:
                break
        return closest_location

    def calculate_euclidean_error(self, delay_time):
        local_fix_point = self.get_gps_location_delayed(delay_time)
        if self.transformed_path_points and local_fix_point:
            last_point = self.transformed_path_points[-1]
            error = np.sqrt((last_point[0] - local_fix_point[0])**2 + 
                            (last_point[1] - local_fix_point[1])**2)# + 
                            # (last_point[2] - local_fix_point[2])**2)
            return error
        return None

    def update_plot(self, frame):
        self.ax_time.clear()
        self.ax_box.clear()

        for delay, errors in self.euclidean_errors.items():
            error = self.calculate_euclidean_error(delay)
            if error is not None and error < 20:
                errors.append(error)
                if len(errors) > 100:  # Maintain a sliding window
                    errors.pop(0)
            
            # Plot error line
            line, = self.ax_time.plot(errors, label=f'{delay} s', linestyle='-')
            color = line.get_color()  # Get the color of the current line to use for the average line
            
            # # Calculate and plot average line
            # valid_errors = [e for e in errors if e < 20]
            # if valid_errors:
            #     avg_error = sum(valid_errors) / len(valid_errors)
            #     self.ax_time.axhline(y=avg_error, color=color, linestyle=':', label=f'{delay} s avg')

        self.ax_time.set_ylim(0, 20)
        self.ax_time.set_title('Time Plot of Successfull fits with Multiple Delays')
        self.ax_time.set_xlabel('Time')
        self.ax_time.set_ylabel('Error [m]')
        self.ax_time.grid(True)
        self.ax_time.legend()

        self.ax_box.set_title('Box Plot')
        self.ax_box.boxplot([errors for errors in self.euclidean_errors.values()], labels=[f'{delay} s' for delay in self.delays])
        

    def shutdown_callback(self):
        plt.close()

if __name__ == '__main__':
    plotter = EuclideanErrorPlotter()
    plt.show()
    rospy.spin()
