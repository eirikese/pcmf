// FILE: gicp_fitting_tools.hpp
#pragma once


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
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/gicp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/search/kdtree.h>
#include <Eigen/Core>
#include <iostream>

#include "pcmf/pcmf_tools.hpp"

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointXYZ PointXYZ;
typedef std::vector<PointXYZ> Line;

void filterFarPoints(const PointCloud::Ptr& slam_cloud, const PointCloud::Ptr& reference_map, PointCloud::Ptr& filtered_map, double max_distance) {
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(slam_cloud);

    for (const auto& point : *reference_map) {
        std::vector<int> indices(1);
        std::vector<float> sqr_distances(1);

        // If the nearest neighbor is within the max_distance, add the point to the filtered_map
        if (kdtree.nearestKSearch(point, 1, indices, sqr_distances) > 0) {
            if (sqr_distances[0] <= max_distance * max_distance) {
                filtered_map->push_back(point);
            }
        }
    }
    filtered_map->header = reference_map->header;
}

// ORIGINAL G-ICP
void alignGicp( PointCloud::Ptr& slam_cloud_transformed, const PointCloud::Ptr& reference_map, std::shared_ptr<Eigen::Matrix4f> transformation_matrix, std::shared_ptr<float> gicp_rmse, double gicp_corresp_dist) {

    if (reference_map->size() < 100) {
        ROS_WARN("Not enough correspondence points for G-ICP");
        return;
    }

    // G-ICP
    PointCloud::Ptr output_cloud(new PointCloud);
    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
    gicp.setInputSource(slam_cloud_transformed);
    // gicp.setInputTarget(reference_map_filtered);
    gicp.setInputTarget(reference_map);
    gicp.setMaximumIterations(1000);
    gicp.setTransformationEpsilon(1e-8); // 1e-8
    gicp.setMaxCorrespondenceDistance(gicp_corresp_dist); // 5 Adjust as needed
    gicp.setEuclideanFitnessEpsilon(1e-8); // 1e-8
    gicp.setRANSACIterations(1000);
    // gicp.setUseReciprocalCorrespondences(true); // means that the correspondence relationship is bi-directional
    gicp.setRANSACOutlierRejectionThreshold(30.0); // 3.0


    gicp.align(*output_cloud);
    if (gicp.hasConverged()) {
        *gicp_rmse = sqrt(gicp.getFitnessScore());
        *transformation_matrix = *transformation_matrix * gicp.getFinalTransformation();
        ROS_INFO("G-ICP transform found, cloud RMSE: %f", *gicp_rmse);
    } else {
        ROS_WARN("G-ICP failed to converge");
    }
}


void scalePointCloud(PointCloud::Ptr& cloud, float scale_factor) {
    for (auto& point : *cloud) {
        point.x *= scale_factor;
        point.y *= scale_factor;
        point.z *= scale_factor;
    }
}
// SCALED G-ICP
void alignGicp_scaled(PointCloud::Ptr& slam_cloud_transformed, const PointCloud::Ptr& reference_map, std::shared_ptr<Eigen::Matrix4f> transformation_matrix, std::shared_ptr<float> gicp_rmse, double gicp_corresp_dist = 10.0) {
    if (reference_map->size() < 100) {
        ROS_WARN("Not enough correspondence points for G-ICP");
        return;
    }

    int scale_factor = 10000;

    // Scale down the clouds
    PointCloud::Ptr scaled_slam_cloud(new PointCloud(*slam_cloud_transformed));
    PointCloud::Ptr scaled_reference_map(new PointCloud(*reference_map));
    scalePointCloud(scaled_slam_cloud, 1/scale_factor);
    scalePointCloud(scaled_reference_map, 1/scale_factor);

    // G-ICP
    PointCloud::Ptr output_cloud(new PointCloud);
    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
    gicp.setInputSource(scaled_slam_cloud);
    gicp.setInputTarget(scaled_reference_map);
    gicp.setMaximumIterations(1000);
    gicp.setTransformationEpsilon(1e-8);
    // gicp.setMaxCorrespondenceDistance(gicp_corresp_dist);
    gicp.setMaxCorrespondenceDistance(0.1);
    gicp.setEuclideanFitnessEpsilon(1e-8);
    gicp.setRANSACIterations(1000);
    gicp.setRANSACOutlierRejectionThreshold(0.1);

    gicp.align(*output_cloud);
    if (gicp.hasConverged()) {
        *gicp_rmse = sqrt(gicp.getFitnessScore());
        Eigen::Matrix4f scaled_up_transformation = Eigen::Matrix4f::Identity();
        scaled_up_transformation.block<3,3>(0,0) = gicp.getFinalTransformation().block<3,3>(0,0);
        scaled_up_transformation.block<3,1>(0,3) = gicp.getFinalTransformation().block<3,1>(0,3) * scale_factor;
        *transformation_matrix = *transformation_matrix * scaled_up_transformation;
        ROS_INFO("G-ICP transform found, cloud RMSE: %f", *gicp_rmse);
    } else {
        ROS_WARN("G-ICP failed to converge");
    }
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////


typedef pcl::PointXYZ PointT;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;

void computeSurfaceNormals(const PointCloud::Ptr &cloud, pcl::PointCloud<pcl::Normal>::Ptr &normals) {
    pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    ne.setSearchMethod(tree);
    ne.setKSearch(20);
    ne.compute(*normals);
}

PointCloudWithNormals::Ptr concatenatePointsAndNormals(const PointCloud::Ptr &points, const pcl::PointCloud<pcl::Normal>::Ptr &normals) {
    PointCloudWithNormals::Ptr points_with_normals(new PointCloudWithNormals);
    pcl::concatenateFields(*points, *normals, *points_with_normals);
    return points_with_normals;
}

void alignGicp_with_normals( PointCloud::Ptr& slam_cloud_transformed, const PointCloud::Ptr& reference_map, std::shared_ptr<Eigen::Matrix4f> transformation_matrix, std::shared_ptr<float> gicp_rmse, double gicp_corresp_dist) {
    // Compute normals for both source and target point clouds
    pcl::PointCloud<pcl::Normal>::Ptr source_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr target_normals(new pcl::PointCloud<pcl::Normal>);
    computeSurfaceNormals(slam_cloud_transformed, source_normals);
    computeSurfaceNormals(reference_map, target_normals);

    // Concatenate points and normals
    PointCloudWithNormals::Ptr src_with_normals = concatenatePointsAndNormals(slam_cloud_transformed, source_normals);
    PointCloudWithNormals::Ptr tgt_with_normals = concatenatePointsAndNormals(reference_map, target_normals);

    // Set up the GICP algorithm
    pcl::GeneralizedIterativeClosestPoint<PointNormalT, PointNormalT> gicp;
    gicp.setMaximumIterations(1000);
    gicp.setTransformationEpsilon(1e-8);
    gicp.setMaxCorrespondenceDistance(gicp_corresp_dist);
    gicp.setEuclideanFitnessEpsilon(1e-8);
    gicp.setRANSACIterations(1000);
    gicp.setRANSACOutlierRejectionThreshold(30.0);

    gicp.setInputSource(src_with_normals);
    gicp.setInputTarget(tgt_with_normals);

    // Perform alignment
    PointCloudWithNormals output;
    gicp.align(output);

    // Extract results
    if (gicp.hasConverged()) {
        *gicp_rmse = std::sqrt(gicp.getFitnessScore());
        *transformation_matrix = *transformation_matrix * gicp.getFinalTransformation();
        std::cout << "G-ICP transform found, cloud RMSE: " << *gicp_rmse << std::endl;
    } else {
        std::cout << "G-ICP failed to converge" << std::endl;
    }
}
