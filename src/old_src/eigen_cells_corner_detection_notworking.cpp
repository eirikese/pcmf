#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/eigen.h>
#include <pcl/common/centroid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Dense>
#include <map>
#include <tuple>
#include <vector>
#include <limits>
#include <visualization_msgs/Marker.h>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
double grid_size = 15.0; // Size of each cell in the grid
const float min_angle = 60.0 * M_PI / 180.0; // for corner detection
const float max_angle = 120.0 * M_PI / 180.0; // for corner detection
const int min_points = 100; // Minimum number of points in a cell to be considered for corner detection
const int voxel_grid_size = 1; // Size of voxel grid filter
const float towards_angle = 180.0 * M_PI / 180.0; // for corner cell classification

class ICPNode {
public:
    ICPNode() : nh_("~") {
        map_points_sub_ = nh_.subscribe("/map_points_processed", 1, &ICPNode::mapPointsCallback, this);
        // grid_cells_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("grid_cells", 1); // publish
        corners_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("detected_corners", 1);
    }

    void mapPointsCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        PointCloud::Ptr cloud(new PointCloud);
        pcl::fromROSMsg(*msg, *cloud);

        pcl::PointCloud<pcl::PointXYZI>::Ptr corners(new pcl::PointCloud<pcl::PointXYZI>);
        detectCorners(cloud, corners);

        ROS_INFO("Number of corners detected: %lu", corners->size());

        sensor_msgs::PointCloud2 corners_msg;
        pcl::toROSMsg(*corners, corners_msg);
        corners_msg.header.frame_id = msg->header.frame_id;
        corners_msg.header.stamp = ros::Time::now();
        corners_pub_.publish(corners_msg);
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber map_points_sub_;
    ros::Publisher corners_pub_;

    std::map<std::tuple<int, int>, PointCloud::Ptr> cells;
    std::map<std::tuple<int, int>, Eigen::Vector3f> cell_eigenvectors;
    std::map<std::tuple<int, int>, Eigen::Vector4f> cell_centroids; // Map for storing centroids

    void detectCorners(const PointCloud::Ptr& cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr& corners) {
        for (auto& point : cloud->points) {
            point.z = 0;
        }

        dividePointCloudIntoCells(cloud);
        computeCellEigenvectors();

        identifyCorners(corners);
    }

    void dividePointCloudIntoCells(const PointCloud::Ptr& cloud) {
        cells.clear();
        for (const auto& point : cloud->points) {
            int x = static_cast<int>(std::floor(point.x / grid_size));
            int y = static_cast<int>(std::floor(point.y / grid_size));
            std::tuple<int, int> cell_index(x, y);
            if (cells.find(cell_index) == cells.end()) {
                cells[cell_index] = PointCloud::Ptr(new PointCloud);
            }
            cells[cell_index]->push_back(point);
        }
    }

    void computeCellEigenvectors() {
        cell_eigenvectors.clear();
        cell_centroids.clear();
        for (const auto& cell : cells) {
            if (cell.second->size() >= min_points) {
                Eigen::Vector4f centroid;
                pcl::compute3DCentroid(*cell.second, centroid);
                cell_centroids[cell.first] = centroid;

                Eigen::Vector3f eigenvector = computeMainEigenvector(cell.second);
                cell_eigenvectors[cell.first] = eigenvector;
            }
        }
    }

    Eigen::Vector3f computeMainEigenvector(PointCloud::Ptr cell_points) {
        Eigen::Matrix3f covariance_matrix;
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cell_points, centroid);
        pcl::computeCovarianceMatrixNormalized(*cell_points, centroid, covariance_matrix);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance_matrix, Eigen::ComputeEigenvectors);
        Eigen::Matrix3f eigenvectors = eigen_solver.eigenvectors();
        return eigenvectors.col(2);
    }

    void identifyCorners(pcl::PointCloud<pcl::PointXYZI>::Ptr& corners) {
        for (const auto& cell : cell_eigenvectors) {
            // rewrite for cpp14 auto [cell_x, cell_y] = cell.first; 
            int cell_x = std::get<0>(cell.first);
            int cell_y = std::get<1>(cell.first);
            

            Eigen::Vector3f current_eigenvector = cell.second;
            Eigen::Vector4f current_centroid = cell_centroids[{cell_x, cell_y}];

            std::vector<std::tuple<int, int>> towards_cells;

            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    if (dx == 0 && dy == 0) continue;

                    std::tuple<int, int> neighbor_index(cell_x + dx, cell_y + dy);

                    auto neighbor_it = cell_eigenvectors.find(neighbor_index);
                    if (neighbor_it != cell_eigenvectors.end()) {
                        Eigen::Vector3f neighbor_vector = neighbor_it->second;
                        Eigen::Vector4f neighbor_centroid = cell_centroids[neighbor_index];
                        
                        Eigen::Vector3f towards_vector = (current_centroid.head<3>() - neighbor_centroid.head<3>()).normalized();

                        // Calculate the cosine of the angle
                        float cos_angle = neighbor_vector.dot(towards_vector) / (neighbor_vector.norm() * towards_vector.norm());
                        ROS_INFO("Cosine of angle: %f", cos_angle * 180.0 / M_PI);

                        // Check if the angle between the neighbor eigenvector and the towards_vector is within the threshold
                        // Using absolute value to allow vectors pointing both ways
                        // if (std::abs(cos_angle) >= std::cos(towards_angle) && std::abs(cos_angle) <= std::cos(max_angle)) {
                        //     towards_cells.push_back(neighbor_index);
                        // }
                        towards_cells.push_back(neighbor_index);
                    }
                }
            }

            if (towards_cells.size() >= 2) {
                Eigen::Vector3f crossing_point = calculateCrossingPoint(towards_cells);

                pcl::PointXYZI corner_point;
                corner_point.x = crossing_point[0];
                corner_point.y = crossing_point[1];
                corner_point.z = crossing_point[2];

                // check if corner point lies within the current cell, if yes, add it to the list of corners
                if (corner_point.x >= cell_x * grid_size && corner_point.x <= (cell_x + 1) * grid_size &&
                    corner_point.y >= cell_y * grid_size && corner_point.y <= (cell_y + 1) * grid_size) {
                    corner_point.intensity = 1.0;
                    corners->push_back(corner_point);
                }
            }
        }
    }

    Eigen::Vector3f calculateCrossingPoint(const std::vector<std::tuple<int, int>>& cell_indices) {
        std::vector<std::pair<float, float>> lines;

        for (const auto& index : cell_indices) {
            auto eigenvector_it = cell_eigenvectors.find(index);
            auto centroid_it = cell_centroids.find(index);

            if (eigenvector_it != cell_eigenvectors.end() && centroid_it != cell_centroids.end()) {
                Eigen::Vector3f eigenvector = eigenvector_it->second;
                Eigen::Vector4f centroid = centroid_it->second;

                float dx = eigenvector[0];
                float dy = eigenvector[1];

                float m = (dx != 0) ? dy / dx : std::numeric_limits<float>::infinity();
                float c = centroid[1] - m * centroid[0];

                lines.push_back({m, c});
            }
        }

        if (lines.size() < 2) {
            return Eigen::Vector3f(0, 0, 0);
        }

        float sum_x = 0, sum_y = 0;
        int count = 0;

        for (size_t i = 0; i < lines.size(); ++i) {
            for (size_t j = i + 1; j < lines.size(); ++j) {
                auto [m1, c1] = lines[i];
                auto [m2, c2] = lines[j];

                if (m1 != m2) {
                    float x = (c2 - c1) / (m1 - m2);
                    float y = m1 * x + c1;
                    sum_x += x;
                    sum_y += y;
                    ++count;
                }
            }
        }

        if (count == 0) {
            return Eigen::Vector3f(0, 0, 0);
        }

        float avg_x = sum_x / count;
        float avg_y = sum_y / count;

        return Eigen::Vector3f(avg_x, avg_y, 0);
    }
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "icp_pcmf_node");
    ICPNode icp_node;
    ros::spin();
    return 0;
}
