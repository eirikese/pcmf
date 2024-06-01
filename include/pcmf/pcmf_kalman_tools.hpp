// File: pcmf_kalman_tools.hpp

#pragma once
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <ros/ros.h>

// Original EKF
void TransformationEKF(std::shared_ptr<std::vector<Eigen::Matrix4f>>& transformation_matrix_history, std::shared_ptr<Eigen::Matrix4f>& estimated_transformation_matrix, std::shared_ptr<Eigen::Matrix4f>& ekf_transformation_matrix) {
    // Define the state vector for x, y translation and z rotation: [x, y, theta]
    Eigen::Vector3f state; 
    // Initialize state with the latest transformation if available
    if (!transformation_matrix_history->empty()) {
        // Convert the latest transformation matrix to translation and Euler angles
        Eigen::Matrix4f& latest_transformation = transformation_matrix_history->back();
        Eigen::Vector3f translation = latest_transformation.block<3,1>(0, 3);
        float theta = std::atan2(latest_transformation(1, 0), latest_transformation(0, 0)); // Z rotation
        state << translation(0), translation(1), theta;
    }

    // Define the state transition model matrix (A) for [x, y, theta]
    Eigen::Matrix3f A = Eigen::Matrix3f::Identity(); // Assuming simple model where the next state is equal to the current state
    
    // Define the process noise covariance matrix (Q) for [x, y, theta], larger means more uncertainty in the process
    Eigen::Matrix3f Q = Eigen::Matrix3f::Identity();
    Q *= 1; //  0.0001

    // Define the measurement model matrix (H) for [x, y, theta], larger means more trust in the measurement
    Eigen::Matrix3f H = Eigen::Matrix3f::Identity(); // Direct measurement of the state

    // Define the measurement noise covariance matrix (R) for [x, y, theta], larger means more uncertainty in the measurement
    Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
    R *= 100000; // Assuming larger measurement noise than process noise = 0.1

    // Initialize the covariance of the state estimate (P) for [x, y, theta], larger means more uncertainty in the initial state
    Eigen::Matrix3f P = Eigen::Matrix3f::Identity();
    P *= 1000; // Assuming large initial uncertainty, 10000 smooths to steady state, 5000 last used
    // P(2, 2) = 10000; // Allow more uncertainty in theta

    // For each observed transformation in history, apply EKF update
    for (const Eigen::Matrix4f& observed_transformation : *transformation_matrix_history) {
        // Convert observed transformation to state vector format
        Eigen::Vector3f observed_translation = observed_transformation.block<3,1>(0, 3);
        float observed_theta = std::atan2(observed_transformation(1, 0), observed_transformation(0, 0));
        Eigen::Vector3f measurement;
        measurement << observed_translation(0), observed_translation(1), observed_theta;

        // Prediction step
        Eigen::Vector3f predicted_state = A * state;
        Eigen::Matrix3f predicted_P = A * P * A.transpose() + Q;

        // Measurement update
        Eigen::Vector3f y = measurement - H * predicted_state;
        Eigen::Matrix3f S = H * predicted_P * H.transpose() + R;
        Eigen::Matrix3f K = predicted_P * H.transpose() * S.inverse();
        state = predicted_state + K * y;
        P = (Eigen::Matrix3f::Identity() - K * H) * predicted_P;
    }

    // Convert the final estimated state back to a transformation matrix and update the estimated transformation. Only allow xy translation and z rotation
    *ekf_transformation_matrix = Eigen::Matrix4f::Identity();
    ekf_transformation_matrix->block<3,1>(0, 3) << state(0), state(1), 0; // XY Translation
    Eigen::Matrix3f rotation;
    rotation = Eigen::AngleAxisf(state(2), Eigen::Vector3f::UnitZ()).toRotationMatrix(); // Z Rotation
    ekf_transformation_matrix->block<3,3>(0, 0) = rotation;

    ROS_INFO("Transformation EKF filtered.");
}

// steady state detection threshold
void TransformationEKF_steady(std::shared_ptr<std::vector<Eigen::Matrix4f>>& transformation_matrix_history, std::shared_ptr<Eigen::Matrix4f>& estimated_transformation_matrix, std::shared_ptr<Eigen::Matrix4f>& ekf_transformation_matrix) {
    // Define the state vector for x, y translation and z rotation: [x, y, theta]
    Eigen::Vector3f state; 
    // Initialize state with the latest transformation if available
    if (!transformation_matrix_history->empty()) {
        Eigen::Matrix4f& latest_transformation = transformation_matrix_history->back();
        Eigen::Vector3f translation = latest_transformation.block<3,1>(0, 3);
        float theta = std::atan2(latest_transformation(1, 0), latest_transformation(0, 0)); // Z rotation
        state << translation(0), translation(1), theta;
    }

    // Define constant parameters
    Eigen::Matrix3f A = Eigen::Matrix3f::Identity(); // State transition model
    Eigen::Matrix3f Q = Eigen::Matrix3f::Identity() * 0.0001; // Process noise covariance
    Eigen::Matrix3f H = Eigen::Matrix3f::Identity(); // Measurement model
    Eigen::Matrix3f R = Eigen::Matrix3f::Identity(); // Initial measurement noise covariance
    Eigen::Matrix3f P = Eigen::Matrix3f::Identity() * 5000; // Initial state estimate covariance

    // Steady state detection threshold
    const float steady_state_threshold = 1.0f; // Threshold to define a large change
    bool is_steady_state = false;
    float steady_state_tolerance = 0.1f; // Tolerance to determine if the system is in steady state

    // Process each observed transformation
    for (const Eigen::Matrix4f& observed_transformation : *transformation_matrix_history) {
        Eigen::Vector3f observed_translation = observed_transformation.block<3,1>(0, 3);
        float observed_theta = std::atan2(observed_transformation(1, 0), observed_transformation(0, 0));
        Eigen::Vector3f measurement;
        measurement << observed_translation(0), observed_translation(1), observed_theta;

        // Prediction step
        Eigen::Vector3f predicted_state = A * state;
        Eigen::Matrix3f predicted_P = A * P * A.transpose() + Q;

        // Measurement update
        Eigen::Vector3f innovation = measurement - H * predicted_state;
        float innovation_magnitude = innovation.norm();

        // Adjust R based on the innovation magnitude
        if (is_steady_state && innovation_magnitude > steady_state_threshold) {
            R = Eigen::Matrix3f::Identity() * 100000; // Increase R to decrease the weight of this measurement
        } else {
            R = Eigen::Matrix3f::Identity() * 0.1; // Normal R value when in or close to steady state
        }

        // Update the steady state flag
        is_steady_state = innovation_magnitude < steady_state_tolerance;

        Eigen::Matrix3f S = H * predicted_P * H.transpose() + R;
        Eigen::Matrix3f K = predicted_P * H.transpose() * S.inverse();
        state = predicted_state + K * innovation;
        P = (Eigen::Matrix3f::Identity() - K * H) * predicted_P;
    }

    // Convert the final state back to a transformation matrix
    estimated_transformation_matrix->setIdentity();
    estimated_transformation_matrix->block<3,1>(0, 3) << state(0), state(1), 0;
    estimated_transformation_matrix->block<2,2>(0, 0) << std::cos(state(2)), -std::sin(state(2)), std::sin(state(2)), std::cos(state(2));
    *ekf_transformation_matrix = *estimated_transformation_matrix;
}



void TransformationUKF(std::shared_ptr<std::vector<Eigen::Matrix4f>>& transformation_matrix_history, std::shared_ptr<Eigen::Matrix4f>& estimated_transformation_matrix) {
    // Define the state vector for x, y translation and z rotation: [x, y, theta]
    Eigen::Vector3f state; 
    if (!transformation_matrix_history->empty()) {
        Eigen::Matrix4f& latest_transformation = transformation_matrix_history->back();
        Eigen::Vector3f translation = latest_transformation.block<3,1>(0, 3);
        float theta = std::atan2(latest_transformation(1, 0), latest_transformation(0, 0));
        state << translation(0), translation(1), theta;
    }

    // State dimension
    int n = 3;
    // Sigma point spreading parameter
    double lambda = 3 - n;
    // Define the state covariance matrix (P)
    Eigen::Matrix3f P = Eigen::Matrix3f::Identity();
    P *= 5000; // Initial uncertainty

    // Define process noise covariance matrix (Q)
    Eigen::Matrix3f Q = Eigen::Matrix3f::Identity();
    Q *= 0.0001; // Small process noise

    // Define measurement noise covariance matrix (R)
    Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
    R *= 100000; // Larger measurement noise

    // Create sigma points
    Eigen::Matrix3f A = P * (lambda + n);
    Eigen::LLT<Eigen::Matrix3f> lltOfA(A); // compute the Cholesky decomposition of A
    Eigen::Matrix3f L = lltOfA.matrixL(); // retrieve factor L  in the decomposition

    // Set of sigma points
    std::vector<Eigen::Vector3f> sigma_points;
    sigma_points.push_back(state);
    for (int i = 0; i < n; ++i) {
        sigma_points.push_back(state + L.col(i));
        sigma_points.push_back(state - L.col(i));
    }

    // Predict and update using each sigma point
    for (const Eigen::Matrix4f& observed_transformation : *transformation_matrix_history) {
        Eigen::Vector3f observed_translation = observed_transformation.block<3,1>(0, 3);
        float observed_theta = std::atan2(observed_transformation(1, 0), observed_transformation(0, 0));
        Eigen::Vector3f measurement;
        measurement << observed_translation(0), observed_translation(1), observed_theta;

        // UKF Prediction
        for (auto& sp : sigma_points) {
            sp = sp; // Assuming no dynamic model (identity model)
        }

        // Mean and covariance prediction
        Eigen::Vector3f mean = Eigen::Vector3f::Zero();
        for (const auto& sp : sigma_points) {
            mean += sp;
        }
        mean /= sigma_points.size();

        P = Q; // reset P
        for (const auto& sp : sigma_points) {
            Eigen::Vector3f diff = sp - mean;
            P += diff * diff.transpose();
        }
        P /= sigma_points.size();

        // UKF Update
        Eigen::Vector3f predicted_measurement = mean; // Direct measurement prediction
        Eigen::Matrix3f S = R; // Measurement prediction covariance
        Eigen::Matrix3f K = P * S.inverse(); // Kalman gain
        state = mean + K * (measurement - predicted_measurement);
        P = (Eigen::Matrix3f::Identity() - K) * P;
    }

    // Convert the final estimated state back to a transformation matrix
    *estimated_transformation_matrix = Eigen::Matrix4f::Identity();
    estimated_transformation_matrix->block<3,1>(0, 3) << state(0), state(1), 0;
    Eigen::Matrix3f rotation = Eigen::AngleAxisf(state(2), Eigen::Vector3f::UnitZ()).toRotationMatrix();
    estimated_transformation_matrix->block<3,3>(0, 0) = rotation;

    ROS_INFO("Transformation UKF filtered.");
}
