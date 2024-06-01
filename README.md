# PCMF Master's Thesis
Point Cloud Map Fitting

Master's Thesis

NTNU, Maritime Robotics 2024

Developed for ROS noetic on Ubuntu 20.04

## Description
This repository contains all the code, documentation, and resources for my master's thesis in collaboration with Maritime Robotics. Here, you'll find the algorithms and software developed for robust positioning of uncrewed surface vessels (USVs) in harbor environments, focusing on sensor fusion using GNSS, LiDAR, and Computer Vision. This repo serves as a resource for the master's thesis.

The code for Point Cloud Object Detection is developed using ROS Noetic and C++, on Ubuntu 20.04. The PCL library is used for point cloud processing. The lidars used for testing are Ouster OS1-64/32, with data transfer via ROS pointcloud2 messages.
Relative Positioning using Fiducial Markers utilizes OpenCV and Fiducial markers for detection and relative positioning.

# Requirements
* fast_gcip
* ouster-ros
* hdl_graph_slam
* ndt_omp

# Parameters
All the configurable parameters are listed in *, launch/pcmf, launch/pcmf_bridge*, and *launch/hdl_graph_slam_400_pcmf*

# Launch
~~~
roslaunch pcmf hdl_graph_slam_400_pcmf.launch
roslaunch pcmf pcmf.launch
rosbag play  ... 
~~~
