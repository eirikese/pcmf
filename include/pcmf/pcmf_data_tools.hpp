// File: pcmf_data_tools.hpp
#pragma once
#include <Eigen/Dense>
#include <string>
#include <sstream>
#include <fstream>


struct IterationData {
    double timestamp;
    Eigen::Matrix4f transformation_matrix;
    Eigen::Matrix4f EKF_transformation_matrix;
    double mapPointsCallback_processing_time;
    int number_of_corners_detected;
    float anchor_rmse;
    float gicp_rmse;
    float gps_longitude;
    float gps_latitude;
    float gps_altitude;

    IterationData() : mapPointsCallback_processing_time(0),
                    number_of_corners_detected(0),
                    anchor_rmse(0.0),
                    gicp_rmse(0.0) {}

    std::string toCSVString() const {
        std::stringstream ss;
        ss << timestamp << ","
        << transformation_matrix.format(Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", ",", "", "", "", ""))
        << ","
        << EKF_transformation_matrix.format(Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", ",", "", "", "", ""))
        << "," << mapPointsCallback_processing_time << ","
        << number_of_corners_detected << ","
        << anchor_rmse << ","
        << gicp_rmse << ","
        << gps_longitude << ","
        << gps_latitude << ","
        << gps_altitude;
        return ss.str();
    }
};