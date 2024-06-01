#!/usr/bin/env python

import rospy
from sensor_msgs.msg import NavSatFix
from visualization_msgs.msg import Marker
import pyproj

# Initialize UTM projector
utm_projector = pyproj.Proj(proj='utm', zone=33, ellps='WGS84')

# Base latitude and longitude
base_lat = 63.439274
base_lon = 10.3990245

# Convert base lat and lon to UTM
base_x, base_y = utm_projector(base_lon, base_lat)

def gps_callback(data):
    # Convert GPS coordinates to UTM
    x, y = utm_projector(data.longitude, data.latitude)
    
    # Calculate local coordinates relative to base
    local_x = x - base_x -200
    local_y = y - base_y

    # Create a marker
    marker = Marker()
    marker.header.frame_id = "map"
    marker.type = marker.SPHERE
    marker.action = marker.ADD
    marker.pose.position.x = local_x
    marker.pose.position.y = local_y
    marker.pose.position.z = 0
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.scale.x = 3.0
    marker.scale.y = 3.0
    marker.scale.z = 3.0
    marker.color.a = 1.0
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 1.0

    # add halo as xy plane circle around the marker with 10m radius
    marker_halo = Marker()
    marker_halo.header.frame_id = "map"
    marker_halo.type = marker.CYLINDER
    marker_halo.action = marker.ADD
    marker_halo.pose.position.x = local_x
    marker_halo.pose.position.y = local_y
    marker_halo.pose.position.z = 0
    marker_halo.pose.orientation.x = 0.0
    marker_halo.pose.orientation.y = 0.0
    marker_halo.pose.orientation.z = 0.0
    marker_halo.pose.orientation.w = 1.0
    marker_halo.scale.x = 20.0
    marker_halo.scale.y = 20.0
    marker_halo.scale.z = 0.1
    marker_halo.color.a = 0.3
    marker_halo.color.r = 0.0
    marker_halo.color.g = 1.0
    marker_halo.color.b = 1.0

    # Publish the marker
    marker_pub.publish(marker)
    halo_pub.publish(marker_halo)

if __name__ == '__main__':
    rospy.init_node('gps_to_local_node', anonymous=True)

    # Subscriber to the GPS fix
    # rospy.Subscriber('/gps/fix', NavSatFix, gps_callback)

    # store buffer 3s back in time, publish with 3s delay
    rospy.Subscriber('/gps/fix', NavSatFix, gps_callback, queue_size=1, buff_size=3)
    
    # Publisher for visualization markers
    marker_pub = rospy.Publisher('/gps/local_fix', Marker, queue_size=10)
    halo_pub = rospy.Publisher('/gps/local_fix_halo', Marker, queue_size=10)

    rospy.spin()
