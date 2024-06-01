#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/poisson.h>
#include <pcl/features/normal_3d.h>
#include <pcl_msgs/PolygonMesh.h>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

class ICPNode {
public:
    ICPNode() : nh_("~"), transformation_matrix_(Eigen::Matrix4f::Identity()) {
        // Subscribers
        map_points_sub_ = nh_.subscribe("/map_points_processed", 1, &ICPNode::mapPointsCallback, this);
        path_points_sub_ = nh_.subscribe("/hdl_graph_slam/path_points", 1, &ICPNode::pathPointsCallback, this);
        reference_map_sub_ = nh_.subscribe("/reference_map", 1, &ICPNode::referenceMapCallback, this);

        // Publishers
        transformed_map_pub_ = nh_.advertise<PointCloud>("transformed_map", 1);
        transformed_path_pub_ = nh_.advertise<PointCloud>("transformed_path", 1);
        corners_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("detected_corners", 1);
    }

    void mapPointsCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        PointCloud::Ptr cloud(new PointCloud);
        pcl::fromROSMsg(*msg, *cloud);

        // Downsample the point cloud
        pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
        voxel_grid.setInputCloud(cloud);
        voxel_grid.setLeafSize(3, 3, 10);
        voxel_grid.filter(*cloud);

        // Multiply point cloud in three layers for 3D Harris
        z_multiply(cloud, 5, 3.0);

        // Detect corners (if needed)
        pcl::PointCloud<pcl::PointXYZI>::Ptr slam_corners(new pcl::PointCloud<pcl::PointXYZI>);
        detectCorners(cloud, slam_corners);

        // Convert the corners to a ROS message and publish
        sensor_msgs::PointCloud2 corners_msg;
        pcl::toROSMsg(*slam_corners, corners_msg);
        corners_msg.header.frame_id = msg->header.frame_id; // Use the same frame as the input
        corners_msg.header.stamp = ros::Time::now();
        corners_pub_.publish(corners_msg);

        // ICP on slam cloud and reference map
        if (!reference_map_.empty() && cloud->size() > 0 ){
            // Perform ICP if the reference map is available
            PointCloud::Ptr transformed_cloud(new PointCloud);
            performICP(reference_map_, *cloud, *transformed_cloud);

            // Publish the transformed map
            transformed_map_pub_.publish(transformed_cloud);
        }

    }

    void referenceMapCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        pcl::fromROSMsg(*msg, reference_map_);
    }

    void pathPointsCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        PointCloud::Ptr cloud(new PointCloud);
        pcl::fromROSMsg(*msg, *cloud);

        // Transform path points using the stored transformation matrix
        PointCloud::Ptr transformed_path(new PointCloud);
        pcl::transformPointCloud(*cloud, *transformed_path, transformation_matrix_);

        // Publish the transformed path
        transformed_path_pub_.publish(transformed_path);
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber map_points_sub_;
    ros::Subscriber path_points_sub_;
    ros::Subscriber reference_map_sub_;
    ros::Publisher transformed_map_pub_;
    ros::Publisher transformed_path_pub_;
    ros::Publisher corners_pub_;

    PointCloud reference_map_;
    Eigen::Matrix4f transformation_matrix_;
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp_;

    void performICP(const PointCloud& reference, const PointCloud& input, PointCloud& output) {
        icp_.setInputSource(input.makeShared());
        icp_.setInputTarget(reference.makeShared());
        icp_.align(output);
        transformation_matrix_ = icp_.getFinalTransformation();
    }

    void detectCorners(const PointCloud::Ptr& cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr& corners) {
        pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI> harris;
        harris.setInputCloud(cloud);
        // harris.setNonMaxSuppression(true);
        harris.setThreshold(0.001); // Set an appropriate threshold value, units are in meters
        harris.setRadius(15);    // Set the neighborhood radius for the Harris operator, units are in meters
        harris.compute(*corners);

        // ros info corners detected
        ROS_INFO("Corners detected: %d", corners->size());

        // Only consider corners with angles larger than +- 45 degrees
        // pcl::PointCloud<pcl::PointXYZI>::Ptr corners_filtered(new pcl::PointCloud<pcl::PointXYZI>);
        // for (int i = 0; i < corners->size(); i++) {
        //     pcl::PointXYZI corner = corners->points[i];
        //     float angle = std::atan2(corner.y, corner.x);
        //     if (angle > -0.785 && angle < 0.785) {
        //         corners_filtered->push_back(corner);
        //     }
        // }
        // corners = corners_filtered;
    }

    void z_multiply(const PointCloud::Ptr& cloud, int layers, float layer_height) {
        PointCloud::Ptr cloud_temp(new PointCloud);
        pcl::copyPointCloud(*cloud, *cloud_temp);
        for (int i = 1; i < layers; i++) {
            PointCloud::Ptr cloud_layer(new PointCloud);
            pcl::copyPointCloud(*cloud_temp, *cloud_layer);
            for (int j = 0; j < cloud_layer->size(); j++) {
                cloud_layer->points[j].z += i * layer_height;
            }
            *cloud += *cloud_layer;
        }
    }
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "icp_pcmf_node");
    ICPNode icp_node;
    ros::spin();
    return 0;
}
