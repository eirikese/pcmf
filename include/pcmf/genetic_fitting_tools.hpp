#ifndef GENETIC_FITTING_TOOLS_HPP
#define GENETIC_FITTING_TOOLS_HPP

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <Eigen/Geometry>
#include <vector>
#include <random>
#include <algorithm>
#include <limits>
#include <cmath>

struct TransformationIndividual {
    double translate_x, translate_y, rotate_z;
    double fitness;

    TransformationIndividual() : translate_x(0.0), translate_y(0.0), rotate_z(0.0), fitness(std::numeric_limits<double>::max()) {}

    // Randomize the individual's parameters within the given bounds
    void randomize(double max_translation, double max_rotation) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis_translation(-max_translation, max_translation);
        std::uniform_real_distribution<> dis_rotation(-max_rotation, max_rotation);

        translate_x = dis_translation(gen);
        translate_y = dis_translation(gen);
        rotate_z = dis_rotation(gen);
    }
};

class PointCloudGA {
    std::vector<TransformationIndividual> population;
    size_t populationSize;
    double mutationRate;
    double crossoverRate;
    int maxGenerations;
    std::mt19937 gen;

public:
    PointCloudGA(size_t populationSize, double mutationRate, double crossoverRate, int maxGenerations)
        : populationSize(populationSize), mutationRate(mutationRate), crossoverRate(crossoverRate), maxGenerations(maxGenerations), gen(std::random_device{}()) {}

    void initializePopulation() {
        population.clear();
        for (size_t i = 0; i < populationSize; ++i) {
            TransformationIndividual individual;
            individual.randomize(10.0, M_PI);  // Randomize within reasonable bounds
            population.push_back(individual);
        }
    }

    void evaluateFitness(pcl::PointCloud<pcl::PointXYZ>::Ptr& slam_corners,
                         const pcl::PointCloud<pcl::PointXYZ>& reference_corners) {
        for (auto& individual : population) {
            Eigen::Affine3f transform = Eigen::Affine3f::Identity();
            transform.translate(Eigen::Vector3f(individual.translate_x, individual.translate_y, 0.0));
            transform.rotate(Eigen::AngleAxisf(individual.rotate_z, Eigen::Vector3f::UnitZ()));

            pcl::PointCloud<pcl::PointXYZ> transformed;
            pcl::transformPointCloud(*slam_corners, transformed, transform);

            double total_distance = 0.0;
            for (const auto& point : transformed) {
                double min_distance = std::numeric_limits<double>::max();
                for (const auto& ref_point : reference_corners) {
                    double distance = pcl::euclideanDistance(point, ref_point);
                    min_distance = std::min(min_distance, distance);
                }
                total_distance += min_distance;
            }

            individual.fitness = total_distance;
        }
    }

    void run(pcl::PointCloud<pcl::PointXYZ>::Ptr& slam_corners,
             const pcl::PointCloud<pcl::PointXYZ>& reference_corners) {
        initializePopulation();
        for (int gen = 0; gen < maxGenerations; ++gen) {
            evaluateFitness(slam_corners, reference_corners);
            crossover();
            mutate();
            // Implement selection and generation replacement
        }
    }

    TransformationIndividual getBestSolution() const {
        return *std::min_element(population.begin(), population.end(),
                                 [](const TransformationIndividual& a, const TransformationIndividual& b) {
                                     return a.fitness < b.fitness;
                                 });
    }

private:
    void crossover() {
        std::uniform_real_distribution<> dis(0.0, 1.0);
        for (size_t i = 0; i < population.size(); i += 2) {
            if (dis(gen) < crossoverRate && i + 1 < population.size()) {
                double temp = population[i].translate_x;
                population[i].translate_x = population[i + 1].translate_x;
                population[i + 1].translate_x = temp;
                // Repeat for other fields (translate_y, rotate_z) as needed
            }
        }
    }

    void mutate() {
        std::uniform_real_distribution<> dis(0.0, 1.0);
        std::normal_distribution<> mutation_dist(0.0, 1.0); // Adjust standard deviation as needed

        for (auto& individual : population) {
            if (dis(gen) < mutationRate) {
                individual.translate_x += mutation_dist(gen);
                individual.translate_y += mutation_dist(gen);
                individual.rotate_z += mutation_dist(gen);
            }
        }
    }

    TransformationIndividual selectParent() {
        std::uniform_int_distribution<> dis(0, population.size() - 1);
        size_t i = dis(gen);
        size_t j = dis(gen);

        return population[i].fitness < population[j].fitness ? population[i] : population[j];
    }
};

void alignCornersGenetic(pcl::PointCloud<pcl::PointXYZ>::Ptr& slam_corners, 
                const pcl::PointCloud<pcl::PointXYZ>& reference_corners, 
                pcl::PointCloud<pcl::PointXYZ>::Ptr& slam_corners_transformed_, 
                Eigen::Matrix4f* transformation_matrix_) {
    // Parameters for the genetic algorithm
    size_t populationSize = 5000;  // Example size, adjust as needed
    double mutationRate = 0.01;   // Example rate, adjust as needed
    double crossoverRate = 0.7;  // Example rate, adjust as needed
    int maxGenerations = 100;    // Example number, adjust as needed

    // Create and run the genetic algorithm
    PointCloudGA ga(populationSize, mutationRate, crossoverRate, maxGenerations);
    ga.run(slam_corners, reference_corners);

    // Get the best solution
    TransformationIndividual best = ga.getBestSolution();
    
    // Apply the best transformation found
    Eigen::Affine3f bestTransform = Eigen::Affine3f::Identity();
    bestTransform.translate(Eigen::Vector3f(best.translate_x, best.translate_y, 0.0));
    bestTransform.rotate(Eigen::AngleAxisf(best.rotate_z, Eigen::Vector3f::UnitZ()));
    pcl::transformPointCloud(*slam_corners, *slam_corners_transformed_, bestTransform);
    *transformation_matrix_ = bestTransform.matrix();
}

#endif // GENETIC_FITTING_TOOLS_HPP
    