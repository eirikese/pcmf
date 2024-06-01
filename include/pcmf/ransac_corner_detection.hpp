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
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/centroid.h>
#include <pcl/common/distances.h>
#include <vector>
#include <algorithm>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>

#include "pcmf/pcmf_tools.hpp"

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointXYZ PointXYZ;
typedef std::vector<PointXYZ> Line; // A line is represented by its two endpoints


// Helper function to find unique close points and merge them
void mergeClosePoints(std::vector<PointXYZ>& points, float threshold, std::vector<PointXYZ>& mergedPoints) {
    std::vector<bool> taken(points.size(), false);
    for (size_t i = 0; i < points.size(); ++i) {
        if (taken[i]) continue;

        PointXYZ current = points[i];
        for (size_t j = i + 1; j < points.size(); ++j) {
            if (taken[j]) continue;

            if (pcl::euclideanDistance(current, points[j]) < threshold) {
                // Merge points by averaging
                current.x = (current.x + points[j].x) / 2;
                current.y = (current.y + points[j].y) / 2;
                taken[j] = true; // Mark as processed
            }
        }
        mergedPoints.push_back(current);
    }
}


// Function to calculate the shortest distance from a point to a line
double pointLineDistance(const PointXYZ& point, const Line& line) {
    Eigen::Vector2d p(point.x, point.y);
    Eigen::Vector2d a(line[0].x, line[0].y);
    Eigen::Vector2d b(line[1].x, line[1].y);
    Eigen::Vector2d ab = b - a;
    Eigen::Vector2d ap = p - a;

    double t = ab.dot(ap) / ab.dot(ab);
    t = std::max(0.0, std::min(1.0, t)); // Clamp t to the range [0, 1]

    Eigen::Vector2d closestPoint = a + t * ab;
    return (closestPoint - p).norm();
}

// Function to check if more than 30% of the line is close to another line
bool isSignificantOverlap(const Line& line1, const Line& line2, double threshold) {
    int samplePointsCount = 100; // Number of points to sample along the line
    int closePoints = 0;

    for (int i = 0; i < samplePointsCount; ++i) {
        // Generate a point on line1
        double t = static_cast<double>(i) / (samplePointsCount - 1);
        PointXYZ sampledPoint;
        sampledPoint.x = line1[0].x + t * (line1[1].x - line1[0].x);
        sampledPoint.y = line1[0].y + t * (line1[1].y - line1[0].y);

        // Check distance from this point to line2
        if (pointLineDistance(sampledPoint, line2) < threshold) {
            closePoints++;
        }
    }

    // Check if more than 30% of the points are within the threshold distance
    return (static_cast<double>(closePoints) / samplePointsCount) > 0.3;
}

// Helper function to calculate the orientation of a line
double calculateOrientation(const Line& line) {
    double dy = line[1].y - line[0].y;
    double dx = line[1].x - line[0].x;
    return std::atan2(dy, dx); // Returns orientation in radians
}

// Helper function to check if two lines are parallel within a threshold
bool areParallel(double orientation1, double orientation2, double angleThresholdDegrees) {
    double angleDifference = std::fabs(orientation1 - orientation2);
    // Normalize angle difference to [0, π]
    angleDifference = std::fmin(angleDifference, M_PI - angleDifference);
    return angleDifference < (angleThresholdDegrees * M_PI / 180.0);
}

// Adjusted mergeLines function to prioritize angle similarity
std::vector<Line> mergeLines(const std::vector<Line>& lines, double angleThresholdDegrees, double distanceThreshold) {
    std::vector<Line> mergedLines;
    std::vector<bool> merged(lines.size(), false);

    for (size_t i = 0; i < lines.size(); ++i) {
        if (merged[i]) continue;

        Line currentLine = lines[i];
        double currentOrientation = calculateOrientation(currentLine);

        for (size_t j = i + 1; j < lines.size(); ++j) {
            if (merged[j]) continue;

            double nextOrientation = calculateOrientation(lines[j]);

            // First check if the lines are roughly parallel to avoid unnecessary distance computations
            if (areParallel(currentOrientation, nextOrientation, angleThresholdDegrees)) {
                // Only if they are parallel do we check for significant overlap
                if (isSignificantOverlap(currentLine, lines[j], distanceThreshold)) {
                    // Merge lines by averaging their endpoints
                    currentLine[0].x = (currentLine[0].x + lines[j][0].x) / 2;
                    currentLine[0].y = (currentLine[0].y + lines[j][0].y) / 2;
                    currentLine[1].x = (currentLine[1].x + lines[j][1].x) / 2;
                    currentLine[1].y = (currentLine[1].y + lines[j][1].y) / 2;
                    merged[j] = true;
                    // Update the orientation after merging
                    currentOrientation = calculateOrientation(currentLine);
                }
            }
        }

        mergedLines.push_back(currentLine);
        merged[i] = true;
    }

    return mergedLines;
}

void detectCornersRansac(const PointCloud::Ptr& cloud, PointCloud::Ptr& corners, int distanceThreshold, int radiusLimitLow, int radiusLimitHigh, int angleThresholdDegrees, int EndpointMergeThreshold, int voxelSize, int lineDistanceThreshold, int minPointsPerLine) {
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::ExtractIndices<pcl::PointXYZ> extract;

    // temp cloud to work with
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    *temp_cloud = *cloud;

    // RANSAC parameters
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_LINE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(1000);
    seg.setDistanceThreshold(distanceThreshold); // 3 distance threshold means how close a point must be to the model in order to be considered an inlier
    seg.setRadiusLimits(radiusLimitLow, radiusLimitHigh); // how to tune this? (min, max) radius limits, how long a line can be

    std::vector<Line> detectedLines;

    while (temp_cloud->points.size() > 100) { // Adjust condition as necessary
        seg.setInputCloud(temp_cloud);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.empty()) break;

        PointCloud::Ptr lineSegment(new PointCloud);
        extract.setInputCloud(temp_cloud);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*lineSegment);

        // Skip line segments with fewer points than the threshold
        // int minPointsPerLine = 20; // 10 Adjust as needed
        if (lineSegment->points.size() < minPointsPerLine) {
            // This line segment is too short, skip it
            extract.setNegative(true);
            extract.filter(*temp_cloud);
            continue;
        }

        if (lineSegment->points.size() > 1) {
            auto [minIt, maxIt] = std::minmax_element(lineSegment->points.begin(), lineSegment->points.end(),
                [](const PointXYZ& a, const PointXYZ& b) { return a.x < b.x; }); // Simplified endpoint detection
            Line line = {*minIt, *maxIt};
            detectedLines.push_back(line);
        }

        extract.setNegative(true);
        extract.filter(*temp_cloud);
    }

    std::vector<Line> mergedLines = mergeLines(detectedLines, angleThresholdDegrees, lineDistanceThreshold);

    // Extract endpoints from merged lines for corner detection
    std::vector<PointXYZ> endPoints;
    for (const auto& line : mergedLines) {
        endPoints.push_back(line[0]);
        endPoints.push_back(line[1]);
    }

    // Merge close endpoints to identify corners
    std::vector<PointXYZ> mergedCorners;
    mergeClosePoints(endPoints, EndpointMergeThreshold, mergedCorners); 

    // Populate the corners point cloud
    for (const auto& point : mergedCorners) {
        corners->push_back(point);
    }
}