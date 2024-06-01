#include <iostream>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <Eigen/Geometry>
#include <pcl/common/pca.h>
#include <std_msgs/Float32.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>


class PointCloudProcessor {
public:
    PointCloudProcessor() : path_points_cloud(new pcl::PointCloud<pcl::PointXYZ>) {
        // Initialize ROS node handle
        ros::NodeHandle nh("~");

        // Get parameters from the parameter server
        if (
            nh.param<float>("z_min", z_min, -1.0) &&
            nh.param<float>("z_max", z_max, 0.0) &&
            nh.param<bool>("use_path_leveling", use_path_leveling, false)
        ) {
            ROS_INFO("Map Processor: Parm load SUCCESS");
        } else {
            ROS_WARN("Map Processor: Parm load FAIL");
        }

        // Subscribers
        map_points_sub = nh.subscribe("/hdl_graph_slam/map_points", 10, &PointCloudProcessor::mapPointsCallback, this);
        markers_sub = nh.subscribe("/hdl_graph_slam/path_points", 10, &PointCloudProcessor::pathPointsCallback, this);
        
        // // filter max and min z values
        // z_minsub = nh.subscribe("/z_min", 1, &PointCloudProcessor::zMinCallback, this);
        // z_maxsub = nh.subscribe("/z_max", 1, &PointCloudProcessor::zMaxCallback, this);

        // Publishers
        leveled_pub = nh.advertise<sensor_msgs::PointCloud2>("/slam_cloud_processed", 10);
        rotation_matrix = Eigen::Matrix4f::Identity(); // Initialize rotation matrix as identity
        num_markers_received = 0;


        // Print message
        // ROS_INFO("Map points processor node initialized with parameters: z_min=%f, z_max=%f", z_min, z_max);
        // ROS_INFO("Other z values can be published to /z_min and /z_max with rqt");
    }

    void mapPointsCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        // Convert the incoming point cloud to PCL format
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        // level point cloud with estimated plane
        if (use_path_leveling) {
            // ROS_WARN("Using path leveling");
            levelPointCloudPathPoints(path_points_cloud, cloud);
        }
        else {
            // ROS_WARN("Using low z leveling");
            levelPointCloudLowZ(cloud);
        }

        // filter z values in separate function
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        filterZvalues(cloud, filtered_cloud, z_min, z_max);

        // Filter noise
        filterNoise(filtered_cloud);

        // Filter voxels
        // filterVoxeling(filtered_cloud, voxel_grid_size); // Uncomment this line to downsample the point cloud

        // Filter proximity
        pcl::PointCloud<pcl::PointXYZ>::Ptr proximity_filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        filterProximity(filtered_cloud, proximity_filtered_cloud, threshold_distance); // Define threshold_distance as needed

        // Convert back to ROS message and publish
        sensor_msgs::PointCloud2 output;
        output.header.frame_id = "map";
        output.header.stamp = ros::Time::now();
        pcl::toROSMsg(*proximity_filtered_cloud, output);
        leveled_pub.publish(output);
    }

    void pathPointsCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        // Convert the incoming point cloud to PCL format and store it
        pcl::fromROSMsg(*msg, *path_points_cloud);
        num_markers_received = path_points_cloud->points.size();

    }

    void zMinCallback(const std_msgs::Float32ConstPtr& msg) {
        z_min = msg->data;
    }

    void zMaxCallback(const std_msgs::Float32ConstPtr& msg) {
        z_max = msg->data;
    }

private:
    void filterZvalues(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& filtered_cloud, float z_min, float z_max) {
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(z_min, z_max);
        pcl::PointCloud<pcl::PointXYZ> temp_cloud;
        pass.filter(temp_cloud);
        *filtered_cloud = temp_cloud;
    }

    // Filter points that are too close to the path points
    void filterProximity(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& filtered_cloud, float threshold) {
        for (const auto& map_point : cloud->points) {
            bool too_close = false;
            for (const auto& path_point : path_points_cloud->points) {
                float distance = euclideanDistance(map_point, path_point);
                if (distance < threshold) {
                    too_close = true;
                    break;
                }
            }
            if (!too_close) {
                filtered_cloud->points.push_back(map_point);
            }
        }
        filtered_cloud->width = filtered_cloud->points.size();
        filtered_cloud->height = 1;
        filtered_cloud->is_dense = true;
        filtered_cloud->header.frame_id = cloud->header.frame_id;
    }

    void filterNoise(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud);
        sor.setMeanK(50); // Number of neighboring points to analyze for each point, higher = remove more points
        sor.setStddevMulThresh(10.0); // Standard deviation multiplier, higher = remove more points
        sor.filter(*cloud);
    }

    void filterVoxeling(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const float voxel_grid_size) {
        pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
        voxel_grid.setInputCloud(cloud);
        voxel_grid.setLeafSize(voxel_grid_size, voxel_grid_size, voxel_grid_size);
        voxel_grid.filter(*cloud);
    }

    float euclideanDistance(const pcl::PointXYZ& point1, const pcl::PointXYZ& point2) {
        float x_diff = point1.x - point2.x;
        float y_diff = point1.y - point2.y;
        return std::sqrt(x_diff * x_diff + y_diff * y_diff);
    }

    void levelPointCloudLowZ(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {

        // Step 1: Sort points by Z value
        std::sort(cloud->points.begin(), cloud->points.end(), [](const pcl::PointXYZ& a, const pcl::PointXYZ& b) {
            return a.z < b.z;
        });

        // Step 2: Sample 10 points with lowest z-values
        pcl::PointCloud<pcl::PointXYZ>::Ptr sampled_points(new pcl::PointCloud<pcl::PointXYZ>);
        size_t interval = std::max(1, static_cast<int>(cloud->points.size() / 10));
        for(size_t i = 0; i < cloud->points.size() && sampled_points->points.size() < 10; i += interval) {
            sampled_points->points.push_back(cloud->points[i]);
        }

        // Step 3: Use sampled points for PCA to find ground plane orientation
        pcl::PCA<pcl::PointXYZ> pca;
        pca.setInputCloud(sampled_points);

        Eigen::Vector3f eigenvalues = pca.getEigenValues().cast<float>();
        Eigen::Matrix3f eigenvectors = pca.getEigenVectors().cast<float>();
        Eigen::Vector3f plane_normal = eigenvectors.col(2); // Assuming the smallest eigenvalue gives the normal

        if (plane_normal[2] < 0) {
            plane_normal = -plane_normal; // Ensuring the normal points upwards
        }

        Eigen::Vector3f z_axis(0, 0, 1);
        Eigen::Quaternionf q;
        q.setFromTwoVectors(plane_normal, z_axis);

        Eigen::Matrix4f rotation_matrix = Eigen::Matrix4f::Identity();
        rotation_matrix.block<3, 3>(0, 0) = q.toRotationMatrix();
        rotation_matrix(2, 3) = 0.0; // Align the plane with z = 0

        // Step 4: Apply the rotation to level the point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr leveled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*cloud, *leveled_cloud, rotation_matrix);
        *cloud = *leveled_cloud;
    }

    void levelPointCloudPathPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr path_points_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
        // Ensure there are enough points for PCA
        if (path_points_cloud->points.size() < 5) {
            ROS_WARN("Skipped PCA leveling (requires at least 5 path points)");
            return;
        }

        // Perform PCA on the stored point cloud
        pcl::PCA<pcl::PointXYZ> pca;
        pca.setInputCloud(path_points_cloud);

        // The normal of the plane is the eigenvector corresponding to the smallest eigenvalue
        Eigen::Vector3f eigenvalues = pca.getEigenValues().cast<float>();
        Eigen::Matrix3f eigenvectors = pca.getEigenVectors().cast<float>();
        Eigen::Vector3f plane_normal = eigenvectors.col(2); // Third eigenvector is the plane's normal

        // Ensure normals are pointing up
        if (plane_normal[2] < 0) {
            plane_normal = -plane_normal;
        }

        // Set up the rotation to align the plane's normal with the z-axis
        Eigen::Vector3f z_axis(0, 0, 1);
        Eigen::Quaternionf q;
        q.setFromTwoVectors(plane_normal, z_axis);

        // Update the transformation matrix
        rotation_matrix = Eigen::Matrix4f::Identity();
        rotation_matrix.block<3, 3>(0, 0) = q.toRotationMatrix();
        rotation_matrix(2, 3) = 0.0; // Align the plane with z = 0

        // Apply the stored rotation matrix to the point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr leveled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*cloud, *leveled_cloud, rotation_matrix);
        *cloud = *leveled_cloud;
    }

    ros::Subscriber map_points_sub;
    ros::Subscriber markers_sub;
    // ros::Subscriber z_minsub;
    // ros::Subscriber z_maxsub;
    ros::Publisher leveled_pub;
    Eigen::Matrix4f rotation_matrix;
    int num_markers_received;

    // Parameters from launch file
    float z_min;
    float z_max;
    bool use_path_leveling;

    pcl::PointCloud<pcl::PointXYZ>::Ptr path_points_cloud;
    float threshold_distance = 3.0; // Define this threshold as per your requirement
    float voxel_grid_size = 1.0; // Define this voxel grid size as per your requirement
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "map_points_processor");
    PointCloudProcessor processor;
    ros::spin();
    return 0;
}
