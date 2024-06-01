#!/usr/bin/env python
import rospy
import csv
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Transform
import tf.transformations
from sensor_msgs.point_cloud2 import read_points

class TopicMonitor:
    def __init__(self):
        rospy.init_node('topic_monitor', anonymous=True)
        self.csv_path = '/home/eirik/lidar_ws/src/pcmf/plot_scripts/eval_data.csv'
        self.data = {'gps/local_fix': None, 'transformation_history': None, 'EKF_transformation_history': None, 'path_points': None, 'camera_trace': None}
        self.fieldnames = ['timestamp', 'gps/local_fix_x', 'gps/local_fix_y', 'transformation_history_x', 'transformation_history_y', 'transformation_history_rotation', 'EKF_transformation_history_x', 'EKF_transformation_history_y', 'EKF_transformation_history_rotation', 'path_points_x', 'path_points_y', 'camera_trace_x', 'camera_trace_y']
        
        with open(self.csv_path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()

        self.sub_gps_local_fix = rospy.Subscriber('/gps/local_fix', Marker, self.callback_gps_local_fix)
        self.sub_transformation_history = rospy.Subscriber('/pcmf_anchor_pivot_node/transformation_history', Transform, self.callback_transformation_history)
        self.sub_ekf_transformation_history = rospy.Subscriber('/pcmf_anchor_pivot_node/EKF_transformation_history', Transform, self.callback_ekf_transformation_history)
        self.sub_path_points = rospy.Subscriber('/pcmf_anchor_pivot_node/transformed_path', PointCloud2, self.callback_path_points)
        self.sub_camera_trace = rospy.Subscriber('/camera_trace', PointCloud2, self.callback_camera_trace)

        rospy.Timer(rospy.Duration(0.1), self.timer_callback)

    def callback_gps_local_fix(self, data):
        position = data.pose.position
        self.data['gps/local_fix'] = {'x': position.x, 'y': position.y}

    def callback_transformation_history(self, data):
        rotation = tf.transformations.euler_from_quaternion([data.rotation.x, data.rotation.y, data.rotation.z, data.rotation.w])
        self.data['transformation_history'] = {'x': data.translation.x, 'y': data.translation.y, 'rotation': rotation[2]}

    def callback_ekf_transformation_history(self, data):
        rotation = tf.transformations.euler_from_quaternion([data.rotation.x, data.rotation.y, data.rotation.z, data.rotation.w])
        self.data['EKF_transformation_history'] = {'x': data.translation.x, 'y': data.translation.y, 'rotation': rotation[2]}

    def callback_path_points(self, data):
        points = list(read_points(data, skip_nans=True))
        if points:
            last_point = points[-1]
            self.data['path_points'] = {'x': last_point[0], 'y': last_point[1]}

    def callback_camera_trace(self, data):
        points = list(read_points(data, skip_nans=True))
        if points:
            last_point = points[-1]
            self.data['camera_trace'] = {'x': last_point[0], 'y': last_point[1]}

    def timer_callback(self, event):
        self.write_to_csv()

    def write_to_csv(self):
        with open(self.csv_path, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            row = {'timestamp': rospy.Time.now().to_sec()}
            for topic, values in self.data.items():
                if values is None:
                    # Log empty fields if no data is available
                    for key in self.fieldnames:
                        if key.startswith(topic):
                            row[key] = ''
                else:
                    # Log available data
                    for key, value in values.items():
                        row[f'{topic}_{key}'] = value
            writer.writerow(row)
        # Reset data to None for the next cycle
        self.data = {key: None for key in self.data}

if __name__ == '__main__':
    tm = TopicMonitor()
    rospy.spin()
