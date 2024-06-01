#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl_conversions/pcl_conversions.h>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZI> PointCloudWithIntensity;

PointCloud::Ptr generatePointCloud() {
    PointCloud::Ptr cloud(new PointCloud());
    
    // Generate synthetic point cloud data (example: simple grid)
    for (float x = -2.0; x <= 2.0; x += 0.1) {
        for (float y = -2.0; y <= 2.0; y += 0.1) {
            pcl::PointXYZ point;
            point.x = x;
            point.y = y;
            point.z = std::sin(x) * std::cos(y);  // Example: Sinusoidal pattern
            cloud->points.push_back(point);
        }
    }
    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = false;

    return cloud;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "harris_example");
    ros::NodeHandle nh;

    ros::Publisher cloud_pub = nh.advertise<PointCloud>("original_cloud", 1);
    ros::Publisher corners_pub = nh.advertise<PointCloudWithIntensity>("harris_corners", 1);

    PointCloud::Ptr cloud = generatePointCloud();
    ROS_INFO("Generated point cloud with %lu points", cloud->points.size());

    // Harris Corner Detection
    pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI> harris;
    harris.setInputCloud(cloud);
    harris.setNonMaxSupression(true);
    harris.setThreshold(0.000001);
    harris.setRadius(1);

    PointCloudWithIntensity::Ptr corners(new PointCloudWithIntensity());
    harris.compute(*corners);
    ROS_INFO("Detected %lu corners", corners->points.size());

    // Convert to ROS message
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud, cloud_msg);
    cloud_msg.header.frame_id = "map";

    sensor_msgs::PointCloud2 corners_msg;
    pcl::toROSMsg(*corners, corners_msg);
    corners_msg.header.frame_id = "map";

    ros::Rate loop_rate(1);
    while (ros::ok()) {
        cloud_msg.header.stamp = ros::Time::now();
        corners_msg.header.stamp = ros::Time::now();

        cloud_pub.publish(cloud_msg);
        corners_pub.publish(corners_msg);

        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
