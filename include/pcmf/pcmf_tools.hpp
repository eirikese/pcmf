// File: pcmf_tools.hpp

#pragma once
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <vector>
#include <Eigen/Geometry>
#include <geometry_msgs/Transform.h>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointXYZ PointXYZ;
typedef std::vector<PointXYZ> Line;

// Downsampling in place
void voxel_downsample(const PointCloud::Ptr& cloud, float leaf_x, float leaf_y, float leaf_z) {
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(leaf_x, leaf_y, leaf_z);
    sor.filter(*cloud);
}

// Downsampling with output cloud
void voxel_downsample(const PointCloud::Ptr& input_cloud, PointCloud::Ptr& output_cloud, float leaf_x, float leaf_y, float leaf_z) {
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(input_cloud);
    sor.setLeafSize(leaf_x, leaf_y, leaf_z);
    sor.filter(*output_cloud);
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

float euclideanDistance(const pcl::PointXYZ& a, const pcl::PointXYZ& b) {
    return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2) + std::pow(a.z - b.z, 2));
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

void extractLargestCluster(PointCloud::Ptr& cloud, int clusterInclusionTolerance = 10, int minClusterSize = 20, int maxClusterSize = 10000) { // remove all clusters except the largest one
    // KD-Tree for searching.
    pcl::search::KdTree<PointXYZ>::Ptr tree(new pcl::search::KdTree<PointXYZ>);
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointXYZ> ec;
    ec.setClusterTolerance(clusterInclusionTolerance);    // 10 m
    ec.setMinClusterSize(minClusterSize);    // 20 Minimum size of a cluster
    ec.setMaxClusterSize(maxClusterSize);  // 10000 Maximum size of a cluster
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    // Find the largest cluster
    size_t largest_cluster_size = 0;
    std::vector<int> largest_cluster_indices;
    for (const auto& indices : cluster_indices) {
        if (indices.indices.size() > largest_cluster_size) {
            largest_cluster_size = indices.indices.size();
            largest_cluster_indices = indices.indices;
        }
    }

    // Extract the largest cluster
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    inliers->indices = largest_cluster_indices;
    pcl::ExtractIndices<PointXYZ> extract;
    PointCloud::Ptr largest_cluster(new PointCloud);
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*largest_cluster);

    // Overwrite the input cloud with the largest cluster
    ROS_INFO("cluster filter before: %lu, after: %lu", cloud->points.size(), largest_cluster->points.size());
    *cloud = *largest_cluster;
}


    void alignCornersGICP(const PointCloud::Ptr& slam_corners,
                        const PointCloud::Ptr& reference_corners,
                        std::shared_ptr<Eigen::Matrix4f> transformation_matrix_) {
        // 1. Align centroids
        Eigen::Vector4f centroid_slam, centroid_reference;
        pcl::compute3DCentroid(*slam_corners, centroid_slam);
        pcl::compute3DCentroid(*reference_corners, centroid_reference);
        Eigen::Vector4f translation = centroid_reference - centroid_slam;

        // Update transformation_matrix_ with the translation
        Eigen::Matrix4f translation_matrix = Eigen::Matrix4f::Identity();
        translation_matrix.block<3,1>(0,3) = translation.head<3>();
        *transformation_matrix_ = translation_matrix * (*transformation_matrix_);

        // 2. Rotate and apply GICP in increments
        float best_score = std::numeric_limits<float>::max();
        Eigen::Matrix4f best_transformation = Eigen::Matrix4f::Identity();

        for(int i = 0; i < 180; ++i) {
            float theta = M_PI * i / 90; // 2 degree increments in radians
            Eigen::Matrix4f rotation_matrix = Eigen::Matrix4f::Identity();
            // Assuming Z-axis rotation
            rotation_matrix(0,0) = cos(theta);
            rotation_matrix(0,1) = -sin(theta);
            rotation_matrix(1,0) = sin(theta);
            rotation_matrix(1,1) = cos(theta);
            rotation_matrix.block<3,1>(0,3) = -centroid_slam.head<3>(); // Translate back to origin
            rotation_matrix = translation_matrix * rotation_matrix; // Apply initial translation
            rotation_matrix.block<3,1>(0,3) += centroid_slam.head<3>(); // Translate back

            // Apply rotation
            PointCloud::Ptr rotated_slam(new PointCloud);
            pcl::transformPointCloud(*slam_corners, *rotated_slam, rotation_matrix);

            // Apply GICP
            pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;

            // tuning parameters
            gicp.setMaximumIterations(1000);
            gicp.setTransformationEpsilon(1e-1);
            gicp.setEuclideanFitnessEpsilon(1e-1);
            gicp.setMaxCorrespondenceDistance(100.0); // 10.0

            // minimum correspondence ratio
            int k_correspondences = std::min(10, static_cast<int>(rotated_slam->size()));
            gicp.setCorrespondenceRandomness(k_correspondences);

            gicp.setInputSource(rotated_slam);
            gicp.setInputTarget(reference_corners);
            PointCloud Final;
            gicp.align(Final);
            float score = gicp.getFitnessScore();

            if(score < best_score) {
                best_score = score;
                best_transformation = gicp.getFinalTransformation();
            }
        }

        // Update transformation_matrix_ with the best transformation found
        *transformation_matrix_ = best_transformation * (*transformation_matrix_);
    }

    // Detect corners using 3D Harris
    void detectCornersHarris(const PointCloud::Ptr& cloud, PointCloud::Ptr& corners) {

        // downsample the cloud
        voxel_downsample(cloud, 3.0, 3.0, 10.0); // 3.0, 3.0, 10.0

        // multiply point cloud in three layers for 3D Harris
        z_multiply(cloud, 10, 3.0); // 5, 3.0

        pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI> harris;
        pcl::PointCloud<pcl::PointXYZI>::Ptr temp_corners(new pcl::PointCloud<pcl::PointXYZI>);
        harris.setInputCloud(cloud);
        harris.setThreshold(0.00001); // 0.001 // larger value detects fewer corners
        harris.setRadius(20);       // 15 ///////////////////////////////////////////////////////////////////////////////////////
        harris.setNonMaxSupression(true); 
        harris.setRefine(true);
        harris.setNumberOfThreads(4);
        harris.setMethod(pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI>::HARRIS);
        harris.setRadiusSearch(20);  // 20
        // harris.setK(0.04);
        harris.setKSearch(0);
        // harris.setNormals(cloud);
        harris.setSearchSurface(cloud);
        harris.compute(*temp_corners);

        for (const auto& point : temp_corners->points) {
            corners->push_back(pcl::PointXYZ(point.x, point.y, point.z));
        }

        ROS_INFO("Corners detected: %d", static_cast<int>(corners->size()));
    }

    // void saveDataToFile() {
    //     ROS_INFO("Initiating data saver...");
        
    //     // default data location
    //     std::string data_location = "/home/eirik/lidar_ws/src/pcmf/pcmf_data/";

    //     // Generate filename with timestamp
    //     std::stringstream filename;
    //     ros::Time now = ros::Time::now();
    //     filename << data_location << "pcmf_data_" << now.sec << ".csv";

    //     std::ofstream file(filename.str().c_str());
    //     if (!file.is_open()) {
    //         ROS_ERROR("Failed to open file: %s", filename.str().c_str());
    //         return;
    //     }

    //     // Write header
    //     file << "Timestamp,Transformation Matrix,EKF Transformation Matrix,Processing Time,Number of Corners Detected,Anchor RMSE,GICP RMSE, GPS Latitude, GPS Longitude, GPS Altitude\n";

    //     // Write data
    //     for (const auto& data : data_history) {
    //         file << data.toCSVString() << "\n";
    //     }

    //     file.close();
    //     ROS_WARN("Data saved to %s", filename.str().c_str());
    // }