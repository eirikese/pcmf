#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Geometry>
#include <limits>
#include <cmath>
#include <geometry_msgs/Transform.h>
#include <pcl/registration/gicp.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/centroid.h>
#include <pcl/common/distances.h>
#include <vector>
#include <algorithm>

#include "pcmf/ransac_corner_detection.hpp"


typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointXYZ PointXYZ;
typedef std::vector<PointXYZ> Line; // A line is represented by its two endpoints

class AnchorFitNode {
public:
    AnchorFitNode() : nh_("~") {
        // Subscribers
        map_points_sub_ = nh_.subscribe("/map_points_processed", 1, &AnchorFitNode::mapPointsCallback, this);
        path_points_sub_ = nh_.subscribe("/hdl_graph_slam/path_points", 1, &AnchorFitNode::pathPointsCallback, this);
        corners_sub_ = nh_.subscribe("/reference_corners", 1, &AnchorFitNode::cornersCallback, this);

        // Publishers
        transformed_map_pub_ = nh_.advertise<PointCloud>("transformed_map", 1);
        transformed_path_pub_ = nh_.advertise<PointCloud>("transformed_path", 1);
        corners_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("detected_corners", 1);
        transformation_pub_ = nh_.advertise<geometry_msgs::Transform>("transformation", 1);
    }

    void mapPointsCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        PointCloud::Ptr cloud(new PointCloud);
        pcl::fromROSMsg(*msg, *cloud);
        ROS_INFO("Cloud size: %d", static_cast<int>(cloud->size()));

        // Detect corners
        PointCloud::Ptr slam_corners(new PointCloud);
        detectCorners(cloud, slam_corners);
        // detectCornersRansac(cloud, slam_corners); // code located in ransa_corner_detection.hpp

        // Convert the corners to a ROS message and publish
        sensor_msgs::PointCloud2 corners_msg;
        pcl::toROSMsg(*slam_corners, corners_msg);
        corners_msg.header.frame_id = msg->header.frame_id;
        corners_msg.header.stamp = ros::Time::now();
        corners_pub_.publish(corners_msg);

        // New alignment approach if reference corners available
        // if (!reference_corners_.empty() && !slam_corners->empty()) {
        if (slam_corners->size() >= 2 && reference_corners_.size() >= 2) {
            PointCloud::Ptr transformed_slam_corners(new PointCloud(*slam_corners));
            float corner_proximity_threshold = 10.0; // Set your threshold
            alignCorners(transformed_slam_corners, reference_corners_, corner_proximity_threshold);
            
            // // Publish the transformed corner points
            // transformed_slam_corners->header.frame_id = msg->header.frame_id;
            // transformed_map_pub_.publish(transformed_slam_corners);

            // transform and publish the cloud
            PointCloud::Ptr transformed_cloud(new PointCloud);
            pcl::transformPointCloud(*cloud, *transformed_cloud, transformation_matrix_);
            transformed_cloud->header.frame_id = msg->header.frame_id;
            transformed_map_pub_.publish(transformed_cloud);

            // Publish the transformation matrix
            geometry_msgs::Transform transform_msg = matrixToTransformMsg(transformation_matrix_);
            transformation_pub_.publish(transform_msg);
        }
    }

    void cornersCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        pcl::fromROSMsg(*msg, reference_corners_);
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
    ros::Subscriber corners_sub_;
    ros::Publisher transformed_map_pub_;
    ros::Publisher transformed_path_pub_;
    ros::Publisher corners_pub_;
    ros::Publisher transformation_pub_;

    PointCloud reference_corners_;
    Eigen::Matrix4f transformation_matrix_;

    float euclideanDistance(const pcl::PointXYZ& a, const pcl::PointXYZ& b) {
        return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2) + std::pow(a.z - b.z, 2));
    }

    pcl::PointXYZ findClosestPointInReference(const pcl::PointXYZ& point, const PointCloud& reference_corners) {
        float min_distance = std::numeric_limits<float>::max();
        pcl::PointXYZ closest_point;
        for (const auto& ref_point : reference_corners.points) {
            float distance = euclideanDistance(point, ref_point);
            if (distance < min_distance) {
                min_distance = distance;
                closest_point = ref_point;
            }
        }
        return closest_point;
    }

    void rotateAroundAnchorToMatch(PointCloud::Ptr& slam_corners, const pcl::PointXYZ& slam_anchor, const pcl::PointXYZ& ref_anchor, float angle) {
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.translate(ref_anchor.getVector3fMap() - slam_anchor.getVector3fMap());
        transform.rotate(Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitZ()));
        pcl::transformPointCloud(*slam_corners, *slam_corners, transform);
    }

    Eigen::Matrix4f calculateCurrentTransformation(const pcl::PointXYZ& slam_anchor, const pcl::PointXYZ& ref_anchor, float angle) {
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.translate(ref_anchor.getVector3fMap() - slam_anchor.getVector3fMap());
        transform.rotate(Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitZ()));
        return transform.matrix();
    }

    bool isAlignmentSatisfactory(const PointCloud::Ptr& slam_corners, const PointCloud& reference_corners, float corner_proximity_threshold, float min_fit_percentage) {
        int total_points = slam_corners->size();
        int fit_points = 0;

        // Count the number of points within the proximity threshold
        for (const auto& slam_point : slam_corners->points) {
            pcl::PointXYZ closest_point = findClosestPointInReference(slam_point, reference_corners);
            if (euclideanDistance(slam_point, closest_point) <= corner_proximity_threshold) {
                fit_points++;
            }
        }

        // Calculate the percentage of points that fit
        float fit_percentage = static_cast<float>(fit_points) / total_points;

        // Return true if the fit percentage is greater than or equal to the minimum required percentage
        return fit_percentage >= min_fit_percentage;
    }

    void scalePointCloud(const PointCloud::Ptr& input_cloud, PointCloud::Ptr& output_cloud, float scale_factor) {
        // Clear the output cloud if it's not empty
        output_cloud->clear();

        // Downscale the coordinates of the points
        for (const auto& input_point : input_cloud->points) {
            pcl::PointXYZ scaled_point;
            scaled_point.x = input_point.x * scale_factor;
            scaled_point.y = input_point.y * scale_factor;
            scaled_point.z = input_point.z * scale_factor;
            output_cloud->push_back(scaled_point);
        }
    }


    // Define a helper function for G-ICP fine-fitting
    void fineFitWithGICP(PointCloud::Ptr& slam_corners, const PointCloud::Ptr& reference_corners) {

        // downscale the point clouds
        int scaling_factor = 1;
        PointCloud::Ptr slam_corners_scaled(new PointCloud);
        PointCloud::Ptr reference_corners_scaled(new PointCloud);
        scalePointCloud(slam_corners, slam_corners_scaled, 1/scaling_factor);
        scalePointCloud(reference_corners, reference_corners_scaled, 1/scaling_factor);

        z_multiply(slam_corners_scaled, 20, 0.1);
        z_multiply(reference_corners_scaled, 20, 0.1);

        // Create a G-ICP object
        pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;

        // Set G-ICP parameters as needed
        gicp.setMaxCorrespondenceDistance(10);
        gicp.setCorrespondenceRandomness(5); 
        gicp.setMaximumIterations(1000);
        gicp.setTransformationEpsilon(1e-2);
        gicp.setEuclideanFitnessEpsilon(1e-2);
        
        // Set input point clouds for G-ICP
        gicp.setInputSource(slam_corners_scaled);
        gicp.setInputTarget(reference_corners_scaled);

        // Create a PointCloud for the aligned result
        PointCloud::Ptr aligned_slam_corners(new PointCloud);

        // Align slam_corners with reference_corners using G-ICP
        gicp.align(*aligned_slam_corners);

        if (gicp.hasConverged()) {
            // Get the final transformation matrix
            Eigen::Matrix4f gicp_transformation = gicp.getFinalTransformation();

            // upscale the transformation matrix
            gicp_transformation(0, 3) *= scaling_factor;
            gicp_transformation(1, 3) *= scaling_factor;
            gicp_transformation(2, 3) *= scaling_factor;

            // Transform slam_corners using the G-ICP transformation
            pcl::transformPointCloud(*slam_corners, *slam_corners, gicp_transformation);

            // Update the overall transformation matrix (if needed)
            transformation_matrix_ = transformation_matrix_ * gicp_transformation;

            ROS_INFO("Fine-fitting using G-ICP applied, translation = (%.2f, %.2f, %.2f), rotation around z = %.2f degrees.",
                gicp_transformation(0, 3), gicp_transformation(1, 3), gicp_transformation(2, 3), std::atan2(gicp_transformation(1, 0), gicp_transformation(0, 0)) * 180 / M_PI);
        } else {
            ROS_WARN("G-ICP failed to converge.");
        }
    }

    // Modify your original alignCorners function to integrate the G-ICP helper
    void alignCorners(PointCloud::Ptr& slam_corners, const PointCloud& reference_corners, float corner_proximity_threshold) {
        Eigen::Matrix4f best_transformation = Eigen::Matrix4f::Identity();
        float best_fit = std::numeric_limits<float>::max();

        for (const auto& slam_anchor : slam_corners->points) {
            for (const auto& ref_anchor : reference_corners.points) {
                for (float angle = 0.0; angle < 2 * M_PI; angle += 0.01) {  // Adjust angle increment as needed
                    PointCloud::Ptr rotated_slam_corners(new PointCloud(*slam_corners));
                    rotateAroundAnchorToMatch(rotated_slam_corners, slam_anchor, ref_anchor, angle);
                    if (isAlignmentSatisfactory(rotated_slam_corners, reference_corners, corner_proximity_threshold, 0.7)) {
                        float current_fit = calculateFit(rotated_slam_corners, reference_corners);
                        if (current_fit < best_fit) {
                            best_fit = current_fit;
                            best_transformation = calculateCurrentTransformation(slam_anchor, ref_anchor, angle);
                        }
                    }
                }
            }
        }

        if (best_fit < std::numeric_limits<float>::max()) {
            // Transform slam_corners using the best_transformation
            pcl::transformPointCloud(*slam_corners, *slam_corners, best_transformation);
            transformation_matrix_ = best_transformation;
            ROS_INFO("Best Anchor rotation fit found and applied.");

            // Fine-fit using G-ICP
            pcl::PointCloud<pcl::PointXYZ>::Ptr reference_corners_xyz(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::copyPointCloud(reference_corners, *reference_corners_xyz);
            fineFitWithGICP(slam_corners, reference_corners_xyz);

            // discard gicp transformation if corner_proximity_threshold is broken
            if (!isAlignmentSatisfactory(slam_corners, reference_corners, corner_proximity_threshold, 0.7)) {
                ROS_WARN("G-ICP transformation discarded.");
                transformation_matrix_ = best_transformation;
            }
        } else {
            ROS_WARN("Failed to find a satisfactory alignment.");
        }
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

    float calculateFit(const PointCloud::Ptr& slam_corners, const PointCloud& reference_corners) {
        float total_distance = 0.0;
        for (const auto& slam_point : slam_corners->points) {
            pcl::PointXYZ closest_point = findClosestPointInReference(slam_point, reference_corners);
            total_distance += euclideanDistance(slam_point, closest_point);
        }
        return total_distance;
    }
    
    // Detect corners using 3D Harris
    void detectCorners(const PointCloud::Ptr& cloud, PointCloud::Ptr& corners) {

        // downsample the cloud
        voxel_downsample(cloud, 3.0, 3.0, 10.0);

        // multiply point cloud in three layers for 3D Harris
        z_multiply(cloud, 5, 3.0);

        pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI> harris;
        pcl::PointCloud<pcl::PointXYZI>::Ptr temp_corners(new pcl::PointCloud<pcl::PointXYZI>);
        harris.setInputCloud(cloud);
        harris.setThreshold(0.001);
        harris.setRadius(15);
        harris.compute(*temp_corners);

        for (const auto& point : temp_corners->points) {
            corners->push_back(pcl::PointXYZ(point.x, point.y, point.z));
        }

        ROS_INFO("Corners detected: %d", static_cast<int>(corners->size()));
    }

    void voxel_downsample(const PointCloud::Ptr& cloud, float leaf_x, float leaf_y, float leaf_z) {
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud);
        sor.setLeafSize(leaf_x, leaf_y, leaf_z);
        sor.filter(*cloud);
    }

    geometry_msgs::Transform matrixToTransformMsg(const Eigen::Matrix4f& matrix) {
        geometry_msgs::Transform transform_msg;
        transform_msg.translation.x = matrix(0, 3);
        transform_msg.translation.y = matrix(1, 3);
        transform_msg.translation.z = matrix(2, 3);

        Eigen::Quaternionf q(Eigen::Matrix3f(matrix.block<3, 3>(0, 0)));
        transform_msg.rotation.x = q.x();
        transform_msg.rotation.y = q.y();
        transform_msg.rotation.z = q.z();
        transform_msg.rotation.w = q.w();

        return transform_msg;
    }

// end of class AnchorFitNode
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "icp_pcmf_node");
    AnchorFitNode icp_node;
    ros::spin();
    return 0;
}
