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
#include "pcmf/genetic_fitting_tools.hpp"
// #include "pcmf/gicp_fitting_tools.hpp"


typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointXYZ PointXYZ;
typedef std::vector<PointXYZ> Line;

class PCMF_Node {
public:
    PCMF_Node() : nh_("~") {
        // Subscribers
        slam_cloud_sub_ = nh_.subscribe("/slam_cloud_processed", 1, &PCMF_Node::mapPointsCallback, this);
        slam_path_sub_ = nh_.subscribe("/hdl_graph_slam/path_points", 1, &PCMF_Node::pathPointsCallback, this);
        ref_corners_sub_ = nh_.subscribe("/reference_corners", 1, &PCMF_Node::cornersCallback, this);
        reference_cloud_sub = nh_.subscribe("/reference_map", 1, &PCMF_Node::refMapPointsCallback, this);

        // Publishers
        transformed_map_pub_ = nh_.advertise<PointCloud>("transformed_map", 1);
        transformed_path_pub_ = nh_.advertise<PointCloud>("transformed_path", 1);
        slam_corners_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("detected_corners", 1);

        // Initialize pointers
        slam_cloud_.reset(new PointCloud);
        slam_path_.reset(new PointCloud);
        slam_corners_.reset(new PointCloud);
        slam_corners_transformed_.reset(new PointCloud);
        slam_cloud_transformed_.reset(new PointCloud);
        slam_path_transformed_.reset(new PointCloud);
        reference_map_.reset(new PointCloud);
        reference_corners_.reset(new PointCloud);

        // Initialize transformation matrix
        transformation_matrix_.reset(new Eigen::Matrix4f(Eigen::Matrix4f::Identity()));
        transformation_matrix_history_.reset(new std::vector<Eigen::Matrix4f>);
    }

    void mapPointsCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        pcl::fromROSMsg(*msg, *slam_cloud_);

        // Detect corners
        detectCornersHarris(slam_cloud_, slam_corners_);
        // detectCornersRansac(cloud, slam_corners); // code located in ransa_corner_detection.hpp

        // Publish the detected corners
        slam_corners_->header.frame_id = msg->header.frame_id;
        slam_corners_pub_.publish(*slam_corners_);

        // New alignment approach if reference corners
        if (slam_corners_->size() >= 2 && reference_corners_->size() >= 2) {
            float corner_proximity_threshold = 10.0; // Set your threshold
            alignCornersAnchor(slam_cloud_transformed_, reference_corners_, corner_proximity_threshold);
            updateAllTransformations();
            
            // fine tune alignment with G-ICP
            // alignGicp(slam_cloud_transformed_, reference_map_, *transformation_matrix_);
            // updateAllTransformations();

            // append acheived transformation in history
            transformation_matrix_history_->push_back(*transformation_matrix_);

            // transform and publish the slam cloud
            pcl::transformPointCloud(*slam_cloud_, *slam_cloud_transformed_, *transformation_matrix_);
            slam_cloud_transformed_->header.frame_id = msg->header.frame_id;
            transformed_map_pub_.publish(slam_cloud_transformed_);

            // transform and publish the path points
            pcl::transformPointCloud(*slam_path_, *slam_path_transformed_, *transformation_matrix_);
            slam_path_transformed_->header.frame_id = msg->header.frame_id;
            transformed_path_pub_.publish(slam_path_transformed_);
        }
    }

    void cornersCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        pcl::fromROSMsg(*msg, *reference_corners_);
    }

    void pathPointsCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        pcl::fromROSMsg(*msg, *slam_path_);
    }

    void refMapPointsCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        pcl::fromROSMsg(*msg, *reference_map_);
    }


private:
    // ROS members
    ros::NodeHandle nh_;
    ros::Subscriber slam_cloud_sub_;
    ros::Subscriber slam_path_sub_;
    ros::Subscriber ref_corners_sub_;
    ros::Subscriber reference_cloud_sub;
    ros::Publisher transformed_map_pub_;
    ros::Publisher transformed_path_pub_;
    ros::Publisher slam_corners_pub_;

    // Transformation matrices
    std::shared_ptr<Eigen::Matrix4f> transformation_matrix_;
    std::shared_ptr<std::vector<Eigen::Matrix4f>> transformation_matrix_history_;

    // Reference map, constant
    PointCloud::Ptr reference_map_;
    PointCloud::Ptr reference_corners_;

    // Slam input
    PointCloud::Ptr slam_path_;
    PointCloud::Ptr slam_cloud_;
    PointCloud::Ptr slam_corners_;

    // Transformed output
    PointCloud::Ptr slam_corners_transformed_;
    PointCloud::Ptr slam_cloud_transformed_;
    PointCloud::Ptr slam_path_transformed_;


    void updateAllTransformations() {
        pcl::transformPointCloud(*slam_corners_, *slam_corners_transformed_, *transformation_matrix_);
        pcl::transformPointCloud(*slam_cloud_, *slam_cloud_transformed_, *transformation_matrix_);
        pcl::transformPointCloud(*slam_path_, *slam_path_transformed_, *transformation_matrix_);
    }

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
        pcl::transformPointCloud(*slam_corners, *slam_corners_transformed_, transform);
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

    // Modify your original alignCorners function to integrate the G-ICP helper
    void alignCornersAnchor(PointCloud::Ptr& slam_corners, const PointCloud::Ptr& reference_corners, float corner_proximity_threshold) {
        Eigen::Matrix4f best_transformation = Eigen::Matrix4f::Identity();
        float best_fit = std::numeric_limits<float>::max();

        for (const auto& slam_anchor : slam_corners->points) {
            for (const auto& ref_anchor : reference_corners->points) {
                for (float angle = 0.0; angle < 2 * M_PI; angle += 0.01) {  // Adjust angle increment as needed
                    PointCloud::Ptr rotated_slam_corners(new PointCloud(*slam_corners));
                    rotateAroundAnchorToMatch(rotated_slam_corners, slam_anchor, ref_anchor, angle);
                    if (isAlignmentSatisfactory(rotated_slam_corners, *reference_corners, corner_proximity_threshold, 0.7)) {
                        float current_fit = calculateFit(rotated_slam_corners, *reference_corners);
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
            *transformation_matrix_ = best_transformation;
            pcl::transformPointCloud(*slam_corners, *slam_corners_transformed_, *transformation_matrix_);
            pcl::transformPointCloud(*slam_cloud_, *slam_cloud_transformed_, *transformation_matrix_);
            ROS_INFO("Best Anchor rotation fit applied.");
        } else {
            ROS_WARN("Anchor fit failed to converge.");
        }
    }

    void alignGicp(PointCloud::Ptr& slam_cloud_transformed, const PointCloud::Ptr& reference_map, Eigen::Matrix4f& transformation_matrix) {
            // Apply G-ICP fine tuning on ref map and transf slam cloud
            PointCloud::Ptr slam_cloud_voxeled(new PointCloud);
            PointCloud::Ptr reference_map_voxeled(new PointCloud);
        
            //downsample with voxeling
            int leaf_size = 1;
            voxel_downsample(slam_cloud_transformed, slam_cloud_voxeled, leaf_size, leaf_size, leaf_size);
            voxel_downsample(reference_map, reference_map_voxeled, leaf_size, leaf_size, leaf_size);

            PointCloud::Ptr output_cloud(new PointCloud);
            pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
            gicp.setInputSource(slam_cloud_voxeled);
            gicp.setInputTarget(reference_map_voxeled);
            gicp.setMaximumIterations(100);
            gicp.setTransformationEpsilon(1e-8);
            gicp.setMaxCorrespondenceDistance(0.1);

            gicp.align(*output_cloud);
            if (gicp.hasConverged()) {
                *transformation_matrix_ = *transformation_matrix_ * gicp.getFinalTransformation();
                pcl::transformPointCloud(*slam_cloud_, *slam_cloud_transformed_, *transformation_matrix_);
                pcl::transformPointCloud(*slam_path_, *slam_path_transformed_, *transformation_matrix_);
                ROS_INFO("G-ICP align applied, score: %f", gicp.getFitnessScore());
            } else {
                ROS_WARN("G-ICP failed to converge");
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
    void detectCornersHarris(const PointCloud::Ptr& cloud, PointCloud::Ptr& corners) {

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

    void voxel_downsample(const PointCloud::Ptr& input_cloud, float leaf_x, float leaf_y, float leaf_z) {
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(input_cloud);
        sor.setLeafSize(leaf_x, leaf_y, leaf_z);
        sor.filter(*input_cloud);
    }

    // Downsampling with output cloud
    void voxel_downsample(const PointCloud::Ptr& input_cloud, PointCloud::Ptr& output_cloud, float leaf_x, float leaf_y, float leaf_z) {
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(input_cloud);
        sor.setLeafSize(leaf_x, leaf_y, leaf_z);
        sor.filter(*output_cloud);
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

// end of class PCMF_Node
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "anchor_pivot_node");
    PCMF_Node pcmf_node;
    ros::spin();
    return 0;
}
