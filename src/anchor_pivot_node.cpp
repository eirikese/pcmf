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
#include <time.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/OccupancyGrid.h>

#include "pcmf/ransac_corner_detection.hpp"
#include "pcmf/pcmf_tools.hpp"
#include "pcmf/gicp_fitting_tools.hpp"
#include "pcmf/pcmf_kalman_tools.hpp"
#include "pcmf/pcmf_data_tools.hpp"

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointXYZ PointXYZ;

class PCMF_Node {
public:
    PCMF_Node() : nh_("~") {
        // Retrieve parameters from the parameter server, if provided
        loadParameters();

        // Subscribers
        slam_cloud_sub_ = nh_.subscribe("/slam_cloud_processed", 1, &PCMF_Node::mapPointsCallback, this);
        slam_path_sub_ = nh_.subscribe("/hdl_graph_slam/path_points", 1, &PCMF_Node::pathPointsCallback, this);
        ref_corners_sub_ = nh_.subscribe("/reference_corners", 1, &PCMF_Node::refCornersCallback, this);
        reference_cloud = nh_.subscribe("/reference_map", 1, &PCMF_Node::refMapPointsCallback, this);
        gps_sub = nh_.subscribe("/gps", 1, &PCMF_Node::gpsCallback, this);

        // Publishers
        transformed_map_pub_ = nh_.advertise<PointCloud>("transformed_map", 1);
        transformed_path_pub_ = nh_.advertise<PointCloud>("transformed_path", 1);
        EKF_transformed_path_pub_ = nh_.advertise<PointCloud>("EKF_transformed_path", 1);
        corners_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("detected_corners", 1);

        // Publishers for transformation history and EKF transformation history vectors, to plot with rqt_plot
        transformation_history_pub_ = nh_.advertise<geometry_msgs::Transform>("transformation_history", 1);
        EKF_transformation_history_pub_ = nh_.advertise<geometry_msgs::Transform>("EKF_transformation_history", 1);

        // Publisher for heatmap
        heatmap_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>("heatmap", 1, true);

        // Initialize pointers
        slam_cloud_.reset(new PointCloud);
        slam_path_.reset(new PointCloud);
        slam_corners_.reset(new PointCloud);
        slam_corners_transformed_.reset(new PointCloud);
        slam_cloud_transformed_.reset(new PointCloud);
        slam_path_transformed_.reset(new PointCloud);
        EKF_slam_path_transformed_.reset(new PointCloud);
        reference_map_.reset(new PointCloud);
        reference_corners_.reset(new PointCloud);
        

        // Initialize transformation matrix
        transformation_matrix_.reset(new Eigen::Matrix4f(Eigen::Matrix4f::Identity()));
        EKF_transformation_matrix_.reset(new Eigen::Matrix4f(Eigen::Matrix4f::Identity()));
        transformation_matrix_history_.reset(new std::vector<Eigen::Matrix4f>);
        EKF_transformation_matrix_history_.reset(new std::vector<Eigen::Matrix4f>);

        // Initialize RMSE pointers
        anchor_rmse_.reset(new float(0.0));
        gicp_rmse_.reset(new float(0.0));

        // Initialize heatmap grid with zeros
        heatmap_grid_ = Eigen::MatrixXf::Zero(grid_height_, grid_width_);

        // Data storage timer
        Node_start_time_ = ros::Time::now().toSec();
    }

    ~PCMF_Node() {
        // saveDataToFile();
    }

    void mapPointsCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        // timer for performance evaluation
        ros::Time start = ros::Time::now();

        PointCloud::Ptr cloud(new PointCloud);
        pcl::fromROSMsg(*msg, *cloud);
        slam_cloud_ = cloud;

        // Detect corners
        PointCloud::Ptr slam_corners(new PointCloud);
        if (useRANSAConCornerDetection) {
            voxel_downsample(cloud, voxelSize, voxelSize, 10.0);
            extractLargestCluster(cloud, clusterInclusionTolerance, minClusterSize, maxClusterSize);
            detectCornersRansac(cloud, slam_corners, distanceThreshold, radiusLimitLow, radiusLimitHigh, angleThresholdDegrees, EndpointMergeThreshold, voxelSize, lineDistanceThreshold, minPointsPerLine); // code located in ransac_corner_detection.hpp
        } else {
            detectCornersHarris(cloud, slam_corners);
        }

        // Store the detected corners
        *slam_corners_ = *slam_corners;

        // Convert the corners to a ROS message and publish
        sensor_msgs::PointCloud2 corners_msg;
        pcl::toROSMsg(*slam_corners, corners_msg);
        corners_msg.header.frame_id = msg->header.frame_id;
        corners_msg.header.stamp = ros::Time::now();
        corners_pub_.publish(corners_msg);

        // sort to prioritize corners based on heatmap
        prioritizeCorners(slam_corners_, grid_cell_size_, grid_width_, grid_height_);
        publishHeatmap(grid_cell_size_, grid_width_, grid_height_);

        // New alignment approach if reference corners
        if (slam_corners_->size() >= 3 && reference_corners_->size() >= 3) {


            // rough alignment with Anchor Pivot on detected corners
            // PointCloud::Ptr transformed_slam_corners(new PointCloud(*slam_corners));
            bool anchor_skipped = false;
            alignCornersAnchor(slam_corners_, reference_corners_, transformation_matrix_, anchor_corner_proximity_threshold, anchor_rmse_, anchor_skipped, checkAllAnchors);
            // alignCornersGICP(slam_corners_, reference_corners_, transformation_matrix_);
            updateAllPoints();

            // print transformation matrix translations from transformation_matrix_
            ROS_INFO("Transformation matrix translations before gicp: %f, %f, %f", transformation_matrix_->block<3,1>(0, 3)(0), transformation_matrix_->block<3,1>(0, 3)(1), transformation_matrix_->block<3,1>(0, 3)(2));

            // fine tune alignment with G-ICP on the whole cloud
            alignGicp(slam_cloud_transformed_, reference_map_, transformation_matrix_, gicp_rmse_, gicp_corresp_dist);
            
            // print transformation matrix translations from transformation_matrix_ after gicp
            ROS_INFO("Transformation matrix translations after gicp: %f, %f, %f", transformation_matrix_->block<3,1>(0, 3)(0), transformation_matrix_->block<3,1>(0, 3)(1), transformation_matrix_->block<3,1>(0, 3)(2));

            updateAllPoints();
            
            // append acheived transformation in history before applying Kalman filter
            transformation_matrix_history_->push_back(*transformation_matrix_);

            // publish the transformed path before applying EKF
            slam_path_transformed_->header.frame_id = msg->header.frame_id;
            transformed_path_pub_.publish(*slam_path_transformed_);

            // kalman filter to smooth the transformation if history larger than 5
            if (transformation_matrix_history_->size() > 5 && enable_pose_EKF) {
                TransformationEKF(transformation_matrix_history_, transformation_matrix_, EKF_transformation_matrix_);
                // TransformationEKF_steady(transformation_matrix_history_, transformation_matrix_, EKF_transformation_matrix_);
                // TransformationUKF(transformation_matrix_history_, transformation_matrix_);
            }

            // append the smoothed transformation in history
            EKF_transformation_matrix_history_->push_back(*EKF_transformation_matrix_);
            updateAllPoints();

            // Publish the transformed map and path
            slam_cloud_transformed_->header.frame_id = msg->header.frame_id;
            transformed_map_pub_.publish(*slam_cloud_transformed_);
            EKF_slam_path_transformed_->header.frame_id = msg->header.frame_id;
            EKF_transformed_path_pub_.publish(*EKF_slam_path_transformed_);

            // Publish the transformation history and EKF transformation history
            geometry_msgs::Transform transform_msg = matrixToTransformMsg(transformation_matrix_history_->back());
            transformation_history_pub_.publish(transform_msg);
            geometry_msgs::Transform EKF_transform_msg = matrixToTransformMsg(EKF_transformation_matrix_history_->back());
            EKF_transformation_history_pub_.publish(EKF_transform_msg);
        }

        // timer for performance evaluation
        double processing_time = (ros::Time::now() - start).toSec();
        ROS_INFO("PCMF processing time: %f", processing_time);

    }

    void refCornersCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        pcl::fromROSMsg(*msg, *reference_corners_);
    }

    void pathPointsCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        PointCloud::Ptr cloud(new PointCloud);
        pcl::fromROSMsg(*msg, *cloud);

        // store path points
        slam_path_ = cloud;
    }

    void refMapPointsCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        PointCloud::Ptr cloud(new PointCloud);
        pcl::fromROSMsg(*msg, *cloud);

        // store reference map points
        reference_map_ = cloud;
    }

    void gpsCallback(const sensor_msgs::NavSatFix::ConstPtr& msg) {
        // store gps data
        gps_data = msg;
    }

private:
    // Parameters from launch file
    double anchor_fit_threshold; // default 0.6, ajust in launch file
    double anchor_corner_proximity_threshold; // default 15.0, ajust in launch file
    int distanceThreshold; // default 3, ajust in launch file
    int radiusLimitLow; // default 15, ajust in launch file
    int radiusLimitHigh; // default 500, ajust in launch file
    int angleThresholdDegrees; // default 10, ajust in launch file
    int EndpointMergeThreshold; // default 10, ajust in launch file
    int voxelSize; // default 3, ajust in launch file
    int lineDistanceThreshold; // default 3, ajust in launch file
    bool checkAllAnchors; // default false, ajust in launch file
    bool useRANSAConCornerDetection; // default true, set to false to use 3D Harris
    int clusterInclusionTolerance;
    int minClusterSize;
    int maxClusterSize;
    int minPointsPerLine;
    int grid_width_; // Adjust based on your environment size and desired resolution
    int grid_height_; // Adjust based on your environment size and desired resolution
    double grid_cell_size_; // 10.0 Size of each cell in meters
    double heatmap_decay_rate; // 0.95 Decay rate for the heatmap, adjust as needed
    bool enable_pose_EKF; // default false, set to true to enable EKF
    double gicp_corresp_dist; // default 10.0, ajust in launch file

    // ROS members
    ros::NodeHandle nh_;
    ros::Subscriber slam_cloud_sub_;
    ros::Subscriber slam_path_sub_;
    ros::Subscriber ref_corners_sub_;
    ros::Subscriber reference_cloud;
    ros::Subscriber gps_sub;
    ros::Publisher transformed_map_pub_;
    ros::Publisher transformed_path_pub_;
    ros::Publisher EKF_transformed_path_pub_;
    ros::Publisher corners_pub_;
    ros::Publisher transformation_history_pub_;
    ros::Publisher EKF_transformation_history_pub_;
    ros::Publisher heatmap_pub_;

    // Transformation matrices
    std::shared_ptr<Eigen::Matrix4f> transformation_matrix_;
    std::shared_ptr<Eigen::Matrix4f> EKF_transformation_matrix_;
    std::shared_ptr<std::vector<Eigen::Matrix4f>> transformation_matrix_history_;
    std::shared_ptr<std::vector<Eigen::Matrix4f>> EKF_transformation_matrix_history_;

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
    PointCloud::Ptr EKF_slam_path_transformed_;

    // gps data ptr
    sensor_msgs::NavSatFix::ConstPtr gps_data;

    // Data storage
    std::shared_ptr<float> anchor_rmse_;
    std::shared_ptr<float> gicp_rmse_;
    std::vector<IterationData> data_history;
    double Node_start_time_;

    // Grid for heatmap
    Eigen::MatrixXf heatmap_grid_;

    void updateAllPoints() {
        *slam_cloud_transformed_ = *slam_cloud_;
        *slam_corners_transformed_ = *slam_corners_;
        *slam_path_transformed_ = *slam_path_;
        *EKF_slam_path_transformed_ = *slam_path_;
        pcl::transformPointCloud(*slam_corners_, *slam_corners_transformed_, *transformation_matrix_);
        pcl::transformPointCloud(*slam_cloud_, *slam_cloud_transformed_, *transformation_matrix_);
        pcl::transformPointCloud(*slam_path_, *slam_path_transformed_, *transformation_matrix_);
        pcl::transformPointCloud(*slam_path_, *EKF_slam_path_transformed_, *EKF_transformation_matrix_);
    }

    pcl::PointXYZ findClosestPointInReference(const pcl::PointXYZ& point, const PointCloud::Ptr& reference_corners) {
        float min_distance = std::numeric_limits<float>::max();
        pcl::PointXYZ closest_point;
        for (const auto& ref_point : reference_corners->points) {
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

    bool isAlignmentSatisfactory(const PointCloud::Ptr& slam_corners, const PointCloud::Ptr& reference_corners, float corner_proximity_threshold, float min_fit_percentage) {
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

    // Anchor Pivot algorithm, aligns the corners of the slam cloud with the reference corners
    void alignCornersAnchor(const PointCloud::Ptr& slam_corners, const PointCloud::Ptr& reference_corners, std::shared_ptr<Eigen::Matrix4f> transformation_matrix_, const float corner_proximity_threshold, std::shared_ptr<float> anchor_rmse_, bool& skipped, bool checkAllAnchors) {
        // check if corners align with previous transformation
        PointCloud::Ptr test_slam_corners(new PointCloud(*slam_corners));
        pcl::transformPointCloud(*test_slam_corners, *test_slam_corners, *transformation_matrix_);
        // if(isAlignmentSatisfactory(test_slam_corners, reference_corners, corner_proximity_threshold, anchor_fit_threshold)) { //////////////////////////////////////////////////////////////////////////////////////////
        //     ROS_INFO("Previous alignment satisfactory, skipping Anchor Pivot.");
        //     skipped = true;
        //     return;
        // }

        // Store previous transformation matrix in temporary variable
        Eigen::Matrix4f previous_transformation = *transformation_matrix_;

        Eigen::Matrix4f best_transformation = Eigen::Matrix4f::Identity();
        float best_fit = std::numeric_limits<float>::max();
        for (const auto& slam_anchor : slam_corners->points) {
            for (const auto& ref_anchor : reference_corners->points) {
                for (float angle = 0; angle < 2 * M_PI; angle += 0.01) {  // (0.01 rad = 0.57 deg) Adjust angle increment as needed
                    PointCloud::Ptr rotated_slam_corners(new PointCloud(*slam_corners));
                    rotateAroundAnchorToMatch(rotated_slam_corners, slam_anchor, ref_anchor, angle);
                    if (isAlignmentSatisfactory(rotated_slam_corners, reference_corners, corner_proximity_threshold, anchor_fit_threshold)) {
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
            *transformation_matrix_ = best_transformation;
            ROS_INFO("Best Anchor rotation fit found, corner RMSE: %f", best_fit);
            *anchor_rmse_ = best_fit;
        } else {
            ROS_WARN("Anchor align fail.");
            // reset transformation to prevoius if no satisfactory alignment is found
            // *transformation_matrix_ = Eigen::Matrix4f::Identity();
            *transformation_matrix_ = previous_transformation;
            skipped = true; // Consider skipping further processing if no satisfactory alignment is found
        }
    }

    float calculateFit(const PointCloud::Ptr& slam_corners, const PointCloud::Ptr& reference_corners) {
        float total_distance = 0.0;
        // rms error calculation
        for (const auto& slam_point : slam_corners->points) {
            pcl::PointXYZ closest_point = findClosestPointInReference(slam_point, reference_corners);
            total_distance += euclideanDistance(slam_point, closest_point);
        }
        return total_distance / slam_corners->size();
    }
    
    void updateHeatmapGrid(const PointCloud::Ptr& corners, float grid_cell_size, int grid_width, int grid_height) {
        for (const auto& corner : corners->points) {
            int x_idx = static_cast<int>((corner.x + grid_width * grid_cell_size / 2) / grid_cell_size);
            int y_idx = static_cast<int>((corner.y + grid_height * grid_cell_size / 2) / grid_cell_size);
            if (x_idx >= 0 && x_idx < grid_width && y_idx >= 0 && y_idx < grid_height) {
                heatmap_grid_(y_idx, x_idx) += 1; // Increase the heat value for the cell
            }
        }
        // Decay the heatmap values
        heatmap_grid_ = heatmap_grid_ * heatmap_decay_rate; // Decay factor, adjust as needed
    }

    float getHeatmapValue(const pcl::PointXYZ& corner, float grid_cell_size, int grid_width, int grid_height) {
        int x_idx = static_cast<int>((corner.x + grid_width * grid_cell_size / 2) / grid_cell_size);
        int y_idx = static_cast<int>((corner.y + grid_height * grid_cell_size / 2) / grid_cell_size);
        if (x_idx >= 0 && x_idx < grid_width && y_idx >= 0 && y_idx < grid_height) {
            return heatmap_grid_(y_idx, x_idx);
        }
        return 0;
    }

    void publishHeatmap(float grid_cell_size, int grid_width, int grid_height) {
        nav_msgs::OccupancyGrid grid_msg;
        grid_msg.header.stamp = ros::Time::now();
        grid_msg.header.frame_id = "map"; // Adjust as per your TF frames
        grid_msg.info.resolution = grid_cell_size;
        grid_msg.info.width = grid_width;
        grid_msg.info.height = grid_height;
        grid_msg.info.origin.position.x = -grid_width * grid_cell_size / 2;
        grid_msg.info.origin.position.y = -grid_height * grid_cell_size / 2;
        grid_msg.info.origin.position.z = 0;
        grid_msg.info.origin.orientation.w = 1.0; // No rotation
        grid_msg.data.resize(grid_width * grid_height);
        for (int y = 0; y < grid_height; ++y) {
            for (int x = 0; x < grid_width; ++x) {
                grid_msg.data[y * grid_width + x] = static_cast<int8_t>(std::min(100.0f, heatmap_grid_(y, x))); // Scaling factor to [0,100] range
            }
        }
        heatmap_pub_.publish(grid_msg);
    }

    void prioritizeCorners(PointCloud::Ptr& corners, float grid_cell_size, int grid_width, int grid_height) {
        // First, update the heatmap with the current corners
        updateHeatmapGrid(corners, grid_cell_size, grid_width, grid_height);

        // Sort corners based on heatmap value
        std::sort(corners->points.begin(), corners->points.end(), [this, grid_cell_size, grid_width, grid_height](const pcl::PointXYZ& a, const pcl::PointXYZ& b) -> bool {
            return getHeatmapValue(a, grid_cell_size, grid_width, grid_height) > getHeatmapValue(b, grid_cell_size, grid_width, grid_height);
        });
    }

    void loadParameters() {
        if (
            nh_.param("anchor_fit_threshold", anchor_fit_threshold, 0.6) &&
            nh_.param("anchor_corner_proximity_threshold", anchor_corner_proximity_threshold, 15.0) &&
            nh_.param("distanceThreshold", distanceThreshold, 3) &&
            nh_.param("radiusLimitLow", radiusLimitLow, 15) &&
            nh_.param("radiusLimitHigh", radiusLimitHigh, 500) &&
            nh_.param("angleThresholdDegrees", angleThresholdDegrees, 10)  &&
            nh_.param("EndpointMergeThreshold", EndpointMergeThreshold, 10) &&
            nh_.param("voxelSize", voxelSize, 3) &&
            nh_.param("lineDistanceThreshold", lineDistanceThreshold, 3) &&
            nh_.param("checkAllAnchors", checkAllAnchors, false) &&
            nh_.param("useRANSAConCornerDetection", useRANSAConCornerDetection, true) &&
            nh_.param("clusterInclusionTolerance", clusterInclusionTolerance, 10) &&
            nh_.param("minClusterSize", minClusterSize, 20) &&
            nh_.param("maxClusterSize", maxClusterSize, 10000) &&
            nh_.param("minPointsPerLine", minPointsPerLine, 20) &&
            nh_.param("grid_width", grid_width_, 100) &&
            nh_.param("grid_height", grid_height_, 100) &&
            nh_.param("grid_cell_size", grid_cell_size_, 5.0) &&
            nh_.param("heatmap_decay_rate", heatmap_decay_rate, 0.95) &&
            nh_.param("enable_pose_EKF", enable_pose_EKF, true) &&
            nh_.param("gicp_corresp_dist", gicp_corresp_dist, 10.0)
        ) {
            ROS_INFO("Anchor pivot: param load SUCCESS");
        } else {
            ROS_WARN("Anchor pivot: param load FAIL");
        }
    }

// end of class PCMF_Node
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "anchor_pivot_node");
    PCMF_Node pcmf_node;
    ros::spin();
    return 0;
}